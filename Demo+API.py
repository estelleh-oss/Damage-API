#!/usr/bin/env python3
"""
Damage estimator pipeline using:
- YOLO detection (ultralytics)
- External estimator API (Bearer key) as primary
- Google Generative Language (text-bison) via API key as fallback
- Local Excel lookups for verification/blending
- Real-time USD<->MXN FX lookup with caching
Environment variables are loaded from a local .env (development only).
Do NOT commit .env or any keys to Git.
"""

from dotenv import load_dotenv
load_dotenv()   # must run before any os.environ.get(...) or startup logging

import time, json, os, sys, logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
# keep app logs at INFO; library noise suppressed
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.INFO)


from typing import Optional, Dict, Any, Tuple

import cv2
import pandas as pd
import requests

# optional imports that may not be present in all environments
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import cvzone
except Exception:
    cvzone = None


# --- Config from environment ---
ESTIMATOR_URL = os.getenv("ESTIMATOR_URL", "").rstrip("/")
ESTIMATOR_API_KEY = os.getenv("ESTIMATOR_API_KEY")        # Bearer token for your estimator service (optional)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")            # API key from AI Studio (used for fallback)
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "Weights/best.pt")
LABOR_XLSX = os.getenv("LABOR_XLSX", "FINAL_LaborRate_with_Reclassification_MXN_Thresholds_and_CarSplit.xlsx")
PARTS_XLSX = os.getenv("PARTS_XLSX", "FINAL_PartCost_with_Reclassification_MXN_Thresholds_and_CarSplit.xlsx")
IMAGE_PATH = os.getenv("IMAGE_PATH", "Media/dent1.jpeg")
CAR_MODEL = os.getenv("CAR_MODEL", "Nissan Pathfinder")
CAR_YEAR = int(os.getenv("CAR_YEAR", "2012"))
USD_FX_FALLBACK = float(os.getenv("USD_FX_FALLBACK", "18.5"))
IVA_RATE = float(os.getenv("IVA_RATE", "0.16"))

# --- FX caching ---
_rate_cache = {"rate": None, "ts": 0}
RATE_TTL = 300  # seconds

def _extract_total_from_verified(verified):
    """
    Accepts either:
      - flat dict with "total" key, or
      - nested dict like {"api": {...}, "local": {...}}
    Returns: (total: float, source: str)
    Raises ValueError if no total found.
    """
    if isinstance(verified, dict) and "total" in verified:
        return float(verified["total"]), verified.get("source", "unknown")

    if isinstance(verified, dict):
        # prefer api
        api = verified.get("api")
        if isinstance(api, dict) and "total" in api:
            return float(api["total"]), api.get("source", "api")
        # then local
        local = verified.get("local")
        if isinstance(local, dict) and "total" in local:
            return float(local["total"]), local.get("source", "local")
        # fallback: search any nested dict for a total
        for v in verified.values():
            if isinstance(v, dict) and "total" in v:
                return float(v["total"]), v.get("source", "nested")
    raise ValueError("Estimator result missing 'total' field")

# at top of file (config)
FX_ACCESS_KEY = os.getenv("FX_ACCESS_KEY")   # add to your .env if you have a key

# replace get_usd_mxn_rate with this version
def get_usd_mxn_rate() -> float:
    now = time.time()
    if _rate_cache["rate"] and now - _rate_cache["ts"] < RATE_TTL:
        return _rate_cache["rate"]

    # 3) exchangerate.host live endpoint (usage counted)
    try:
        if FX_ACCESS_KEY:
            r = requests.get(
                "https://api.exchangerate.host/live",
                params={"access_key": FX_ACCESS_KEY, "base": "USD", "symbols": "MXN"},
                timeout=8
            )
            j = r.json()
            usd_mxn_rate = j["quotes"]["USDMXN"]
            r.raise_for_status()
            j = r.json()
            if j.get("success") and "USDMXN" in j.get("quotes", {}):
                rate = float(j["quotes"]["USDMXN"])
                _rate_cache.update({"rate": rate, "ts": now})
                logging.info("FX rate fetched (exchangerate.host live): 1 USD = %.4f MXN", rate)
                return rate

    except Exception as e:
        logging.debug("exchangerate.host live attempt failed: %s", e)

    # 1) open provider (no key)
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=6)
        r.raise_for_status()
        j = r.json()
        if "rates" in j and "MXN" in j["rates"]:
            rate = float(j["rates"]["MXN"])
            _rate_cache.update({"rate": rate, "ts": now})
            logging.info("FX rate fetched (open): 1 USD = %.4f MXN", rate)
            return rate
    except Exception as e:
        logging.debug("Open FX provider failed: %s", e)

    # 2) try apilayer endpoint with header (if key present)
    if FX_ACCESS_KEY:
        try:
            r = requests.get(
                "https://api.apilayer.com/exchangerates_data/latest",
                params={"base": "USD", "symbols": "MXN"},
                headers={"apikey": FX_ACCESS_KEY},
                timeout=8
            )
            r.raise_for_status()
            j = r.json()
            if "quotes" in j and "USDMXN" in j["quotes"]:
                rate = float(j["quotes"]["USDMXN"])
                _rate_cache.update({"rate": rate, "ts": now})
                logging.info("FX rate fetched (apilayer): 1 USD = %.4f MXN", rate)
                return rate
        except Exception as e:
            logging.debug("apilayer attempt failed: %s", e)

    # fallback
    logging.warning("Using fallback FX rate: %.4f", USD_FX_FALLBACK)
    return USD_FX_FALLBACK


USD_to_MXN = get_usd_mxn_rate()
MXN_to_USD = 1.0 / USD_to_MXN

# --- Load local reference tables ---
def _load_tables():
    try:
        labor_df = pd.read_excel(LABOR_XLSX)
        parts_df = pd.read_excel(PARTS_XLSX)
    except Exception as e:
        logging.error("Failed to load Excel reference files: %s", e)
        raise
    def normalize_label(s):
        return str(s).lower().replace(" ", "").replace("-", "")
    labor_df["Reclassified_Type_norm"] = labor_df["Reclassified_Type"].apply(normalize_label)
    parts_df["Reclassified_Type_norm"] = parts_df["Reclassified_Type"].apply(normalize_label)
    return labor_df, parts_df

labor_df, parts_df = _load_tables()

# --- Local lookup ---
def lookup_local_cost(label: str, car_model: str, car_year: int) -> Optional[Dict[str, float]]:
    label_norm = str(label).lower().replace(" ", "").replace("-", "")
    if label_norm == "pillar-dent":
        return None
    if label_norm == "quaterpanel-dent":
        ref_parts = parts_df[
            (parts_df["Reclassified_Type"] == "Bodypanel-Dent") &
            (parts_df["Car_Model_Name"].str.contains(car_model, case=False, na=False)) &
            (parts_df["Car_Year"] == car_year)
        ]
        ref_labor = labor_df[
            (labor_df["Reclassified_Type"] == "Bodypanel-Dent") &
            (labor_df["Car_Model_Name"].str.contains(car_model, case=False, na=False)) &
            (labor_df["Car_Year"] == car_year)
        ]
        if not ref_parts.empty and not ref_labor.empty:
            parts_cost = float(ref_parts["Amount_MXN"].mean()) * 0.25
            labor_rate_mxn = float(ref_labor["Labor (MXN per hr)"].mean()) * 0.25
            labor_rate_usd = labor_rate_mxn * MXN_to_USD
            total = parts_cost + labor_rate_mxn
            return {"parts_cost": parts_cost, "labor_rate": labor_rate_mxn, "total": total}
        return None
    ref_parts = parts_df[
        (parts_df["Reclassified_Type_norm"] == label_norm) &
        (parts_df["Car_Model_Name"].str.contains(car_model, case=False, na=False)) &
        (parts_df["Car_Year"] == car_year)
    ]
    ref_labor = labor_df[
        (labor_df["Reclassified_Type_norm"] == label_norm) &
        (labor_df["Car_Model_Name"].str.contains(car_model, case=False, na=False)) &
        (labor_df["Car_Year"] == car_year)
    ]
    if not ref_parts.empty and not ref_labor.empty:
        parts_cost = float(ref_parts["Amount_MXN"].mean())
        labor_rate_mxn = float(ref_labor["Labor (MXN per hr)"].mean())
        labor_rate_usd = labor_rate_mxn * MXN_to_USD
        total = parts_cost + labor_rate_mxn
        return {"parts_cost": parts_cost, "labor_rate": labor_rate_mxn, "total": total}
    return None

# --- External estimator API (primary) ---
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests, logging, os


def call_estimator_api(label, severity, car_model, car_year, damage_fraction):
    payload = {"label": label, "severity": float(severity), "car_model": car_model,
               "car_year": int(car_year), "damage_fraction": float(damage_fraction)}
    base = (os.environ.get("ESTIMATOR_URL") or "").rstrip("/")
    url = base if base.endswith("/predict") else base + "/predict"
    api_key = os.environ.get("ESTIMATOR_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # session with retries
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    logging.info("Calling estimator at %s auth=%s", url, "bearer" if api_key else "no-key")
    try:
        r = session.post(url, json=payload, headers=headers, timeout=10)
        if r.status_code in (401, 403) and api_key:
            logging.info("Bearer rejected; retrying with x-api-key header")
            r = session.post(url, json=payload, headers={"x-api-key": api_key, "Content-Type":"application/json"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        # ensure required field exists and tag provenance
        if not isinstance(data, dict) or "total" not in data:
            logging.error("Estimator returned unexpected payload: %s", data)
            raise ValueError("Estimator response missing required field 'total'")
        data["source"] = "api"
        logging.debug("API_RAW_RESPONSE:", data)
        logging.info("API_RAW_RESPONSE keys=%s total=%s", list(data.keys()), data.get("total"))
        return data

    except requests.RequestException as exc:
        logging.error("Estimator request failed: %s", exc)
        raise
    except ValueError:
        logging.error("Estimator returned non-JSON response: %s", (r.text or "")[:1000])
        raise
    # normalize numeric fields as you already do
    return data


# --- Google Generative Language fallback using API key (text-bison) ---
def call_google_bison_with_api_key(prompt: str, model: str = "text-bison-001", max_tokens: int = 400) -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateText"
    params = {"key": GOOGLE_API_KEY}
    body = {"prompt": {"text": prompt}, "maxOutputTokens": max_tokens, "temperature": 0.2}
    r = requests.post(url, params=params, json=body, timeout=15)
    r.raise_for_status()
    j = r.json()
    candidates = j.get("candidates", [])
    if not candidates:
        raise RuntimeError("No generation candidates returned")
    return candidates[0].get("content", "")

def estimate_with_google(label: str, box_area: float, img_area: float, labor_rate_mxn: float) -> Tuple[float, Dict[str, Any]]:
    """
    Use Google Generative Language (text-bison) as a fallback estimator.
    Returns (severity, parsed_estimate) where parsed_estimate contains numeric fields
    and a provenance tag 'source': 'google'.
    """
    damage_fraction = box_area / img_area
    damage_fraction = max(0.01, min(damage_fraction, 0.5))
    severity = damage_fraction / 0.5

    prompt = f"""
You are an experienced auto repair estimator in Mexico.
Damage type: {label}
Severity (0-1): {severity:.2f}
Labor rate: {labor_rate_mxn:.2f} MXN/hour
IVA: {IVA_RATE*100:.0f}%

Return a JSON object only with keys:
label, severity, parts_cost, labor_hours, labor_cost, subtotal, iva, total
Numeric values only (no currency symbols).
"""

    # Call the Google API (wrapper defined elsewhere)
    gen = call_google_bison_with_api_key(prompt)

    # Try to extract JSON from the model output
    parsed: Dict[str, Any] = {}
    try:
        first = gen.find("{")
        json_text = gen[first:] if first != -1 else gen
        parsed = json.loads(json_text)
    except Exception:
        # Best-effort line-by-line parse if the model returned a non-strict JSON
        parsed = {}
        for line in gen.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip().strip('"').strip("'")
                v = v.strip().split()[0].replace(",", "")
                try:
                    parsed[k] = float(v)
                except Exception:
                    parsed[k] = v

    # Ensure numeric fields are floats and present
    for k in ["parts_cost", "labor_hours", "labor_cost", "subtotal", "iva", "total"]:
        if k in parsed:
            try:
                parsed[k] = float(parsed[k])
            except Exception:
                parsed[k] = 0.0

    # Fill defaults and provenance
    parsed.setdefault("label", label)
    parsed.setdefault("severity", severity)
    parsed["source"] = "google"

    # Validate required keys exist
    required = {"parts_cost", "labor_hours", "labor_cost", "subtotal", "iva", "total"}
    missing = required - set(k for k in parsed.keys())
    if missing:
        logging.error("Google estimator missing keys: %s; raw_output=%s", missing, gen)
        raise ValueError(f"Google estimator response missing keys: {missing}")

    return severity, parsed

# --- Unified estimator with fallback to Google ---
def estimate_damage_cost_mxn(label: str, box_area: float, img_area: float, labor_rate_mxn: float) -> Tuple[float, Dict[str, Any]]:
    logging.debug("REACHED estimate_damage_cost_mxn", {"label": label, "box_area": box_area, "img_area": img_area})
    logging.info("ENTER estimate_damage_cost_mxn label=%s box_area=%s img_area=%s", label, box_area, img_area)
    damage_fraction = box_area / img_area
    damage_fraction = max(0.01, min(damage_fraction, 0.5))
    severity = damage_fraction / 0.5

    # DEBUG: show what we will call the estimator with
    logging.info("DEBUG_PRINT: estimate_damage_cost_mxn will call estimator with label=%s severity=%.4f df=%.6f",
                 label, severity, damage_fraction)
    print("DEBUG_PRINT: estimate_damage_cost_mxn calling estimator:", {
        "label": label, "severity": severity, "car_model": CAR_MODEL, "car_year": CAR_YEAR, "damage_fraction": damage_fraction
    })

    # try primary estimator API
    # replace the try/except around call_estimator_api in estimate_damage_cost_mxn with this
    try:
        api_resp = call_estimator_api(label, severity, CAR_MODEL, CAR_YEAR, damage_fraction)
        logging.info("Estimator API returned total=%.2f MXN for %s", api_resp.get("total", 0.0), label)
        return severity, api_resp
    except Exception as e:
        logging.exception("Estimator API call raised an exception; falling back.")
        logging.debug("ESTIMATOR_EXCEPTION:", repr(e))
        # continue with your existing fallback logic (Google/local/default)
        try:
            return estimate_with_google(label, box_area, img_area, labor_rate_mxn)
        except Exception as e2:
            logging.error("Google fallback failed: %s", e2)
            local = lookup_local_cost(label, CAR_MODEL, CAR_YEAR)
            if local:
                subtotal = local["total"]
                iva = subtotal * IVA_RATE
                total = subtotal + iva
                return severity, {"parts_cost": local["parts_cost"], "labor_hours": 1.0, "labor_cost": local["labor_rate"],
                                  "subtotal": subtotal, "iva": iva, "total": total}
            parts = 500.0
            labor_cost = labor_rate_mxn * 1.0
            subtotal = parts + labor_cost
            iva = subtotal * IVA_RATE
            total = subtotal + iva
            return severity, {"parts_cost": parts, "labor_hours": 1.0, "labor_cost": labor_cost,
                              "subtotal": subtotal, "iva": iva, "total": total}
        # after you obtain api_resp or google/local fallback
    source = result.get("source", "unknown")
    logging.info("Estimator used: %s | label=%s | total=%.2f MXN", source, label, float(result.get("total", 0.0)))


# --- Verification / blending ---
def verify_estimate(api_data: Dict[str, Any], local_data: Optional[Dict[str, float]], tolerance: float = 0.15):
    if local_data is None:
        return "provisional", api_data
    api_total = float(api_data["total"])
    local_total = float(local_data["total"])
    diff = abs(api_total - local_total) / local_total
    if diff <= tolerance:
        return "accepted", api_data
    elif diff <= 2 * tolerance:
        blended_total = 0.7 * local_total + 0.3 * api_total
        api_data["total"] = blended_total
        return "blended", api_data
    else:
        return "flagged", {"api": api_data, "local": local_data}

# --- YOLO setup ---
if YOLO is None:
    logging.error("ultralytics YOLO not installed. Install with: pip install ultralytics")
else:
    yolo_model = YOLO(YOLO_WEIGHTS)

class_labels = [
    'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
    'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

# --- Main flow ---
def run_image_estimation(image_path: str):
    if YOLO is None:
        raise RuntimeError("YOLO model not available")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w

    # labor rate fallback: average of labor_df converted to MXN
    try:
        labor_rate_mxn = float(labor_df["Labor (MXN per hr)"].mean())
        labor_rate_usd = labor_rate_mxn * MXN_to_USD
    except Exception:
        labor_rate_mxn = 200.0

    logging.info("Using labor rate: %.2f MXN/hr", labor_rate_mxn)

    results = yolo_model(img)

    # Totals and review collection
    total_claim_mxn = 0.0
    flagged_total_mxn = 0.0
    flagged_items = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            box_area = max(1, w * h)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = class_labels[cls] if cls < len(class_labels) else f"class_{cls}"

            if conf <= 0.3:
                continue

            if cvzone:
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)

            # Primary estimator + fallbacks
            severity, api_data = estimate_damage_cost_mxn(label, box_area, img_area, labor_rate_mxn)
            local_data = lookup_local_cost(label, car_model=CAR_MODEL, car_year=CAR_YEAR)
            status, verified_data = verify_estimate(api_data, local_data)

            # Extract total robustly (handles flat and nested shapes)
            try:
                est_total_mxn, total_source = _extract_total_from_verified(verified_data)
            except Exception:
                logging.error("Missing 'total' in verified_data: %s", verified_data)
                logging.exception("Fatal error reading estimated total")
                raise

            # Accumulate according to policy: flagged items are doubtful and not auto-included
            if status in ["accepted", "blended", "provisional"]:
                total_claim_mxn += est_total_mxn
            elif status == "flagged":
                flagged_total_mxn += est_total_mxn
                flagged_items.append({
                    "label": label,
                    "conf": conf,
                    "api_total": verified_data.get("api", {}).get("total") if isinstance(verified_data, dict) else None,
                    "local_total": verified_data.get("local", {}).get("total") if isinstance(verified_data, dict) else None,
                    "chosen_source": total_source
                })

            est_total_usd = est_total_mxn * MXN_to_USD

            if status == "flagged":
                api_total = verified_data.get("api", {}).get("total") if isinstance(verified_data, dict) else None
                local_total = verified_data.get("local", {}).get("total") if isinstance(verified_data, dict) else None
                logging.info("FLAGGED: %s | Conf: %.2f | API: %s MXN | Local: %s",
                             label, conf,
                             f"{api_total:.2f}" if api_total is not None else f"{est_total_mxn:.2f}",
                             f"{local_total:.2f}" if local_total is not None else "n/a")
            else:
                logging.info("%s | Conf: %.2f | %s | Total: %.2f MXN / %.2f USD",
                             label, conf, status, est_total_mxn, est_total_usd)

            # Overlay text: mark flagged items as doubtful
            if status == "flagged":
                overlay_text = f'{label} {conf:.2f} | DOUBTFUL — review required ~{int(est_total_mxn)} MXN'
            else:
                overlay_text = f'{label} {conf:.2f} | {status} ~{int(est_total_mxn)} MXN (~{int(est_total_usd)} USD)'

            if cvzone:
                cvzone.putTextRect(img, overlay_text, (x1, y1 - 10), scale=0.7)
            else:
                cv2.putText(img, overlay_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Summary logs: accepted vs flagged
    logging.info("Estimated TOTAL CLAIM (accepted/blended/provisional): %.2f MXN / %.2f USD",
                 total_claim_mxn, total_claim_mxn * MXN_to_USD)
    logging.info("Flagged items total (doubtful, not included): %.2f MXN / %.2f USD",
                 flagged_total_mxn, flagged_total_mxn * MXN_to_USD)
    if flagged_items:
        logging.info("Flagged items details: %s", flagged_items)
        # optional: export flagged_items for manual review (uncomment to enable)
        # import csv
        # with open("flagged_review.csv","w",newline="") as f:
        #     writer = csv.DictWriter(f, fieldnames=flagged_items[0].keys())
        #     writer.writeheader(); writer.writerows(flagged_items)

    # Display image and wait for 'q' to quit
    cv2.imshow("Image", img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# --- Entrypoint ---
if __name__ == "__main__":
    try:
        run_image_estimation("7692.jpg")
    except Exception as e:
        logging.exception("Fatal error: %s", e)
