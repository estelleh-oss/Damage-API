#!/usr/bin/env python3
"""
Integrated Damage Estimator Pipeline
- Detection: YOLO (Ultralytics)
- Primary Estimation: Local Python API (127.0.0.1:8080)
- Fallback Estimation: Google Generative Language (text-bison)
- Verification: Local Excel tables
- FX: Real-time via exchangerate.host (with caching)
"""

from dotenv import load_dotenv

load_dotenv()  # Load .env variables

import time, json, os, sys, logging
import requests
import cv2
import pandas as pd
import math
from typing import Optional, Dict, Any, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- Optional Imports ---
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    logging.warning("Ultralytics not installed. YOLO detection will fail.")

try:
    import cvzone
except Exception:
    cvzone = None

# --- Configuration ---
# Updated default URL to match your successful test
ESTIMATOR_URL = os.getenv("ESTIMATOR_URL", "http://127.0.0.1:8080/estimate")
ESTIMATOR_API_KEY = os.getenv("ESTIMATOR_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FX_ACCESS_KEY = os.getenv("FX_ACCESS_KEY")

# File paths
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "Weights/best.pt")
LABOR_XLSX = os.getenv("LABOR_XLSX", "FINAL_LaborRate_with_Reclassification_MXN_Thresholds_and_CarSplit.xlsx")
PARTS_XLSX = os.getenv("PARTS_XLSX", "FINAL_PartCost_with_Reclassification_MXN_Thresholds_and_CarSplit.xlsx")
IMAGE_PATH = os.getenv("IMAGE_PATH", "33.jpg")

# Car Settings
# --- Interactive Car Settings ---
# 1. Get defaults from Env or Hardcode
default_model = os.getenv("CAR_MODEL", "Nissan Pathfinder")
default_year = os.getenv("CAR_YEAR", "2012")

# 2. Ask User for Input (Press Enter to use default)
print(f"\n--- Vehicle Details ---")
user_model = input(f"Enter Car Model [Default: {default_model}]: ").strip()
user_year = input(f"Enter Car Year  [Default: {default_year}]: ").strip()

# 3. Set Final Values
CAR_MODEL = user_model if user_model else default_model
try:
    CAR_YEAR = int(user_year) if user_year else int(default_year)
except ValueError:
    logging.warning("Invalid year entered. Using default.")
    CAR_YEAR = int(default_year)

print(f"Using: {CAR_MODEL} ({CAR_YEAR})\n")

REGION = "Mexico"
CURRENCY = "MXN"

# Fallbacks
USD_FX_FALLBACK = float(os.getenv("USD_FX_FALLBACK", "18.5"))
IVA_RATE = float(os.getenv("IVA_RATE", "0.16"))

# --- FX Caching System ---
_rate_cache = {"rate": None, "ts": 0}
RATE_TTL = 300  # seconds


def get_usd_mxn_rate() -> float:
    """Fetches real-time FX rate, prioritizing exchangerate.host"""
    now = time.time()
    if _rate_cache["rate"] and now - _rate_cache["ts"] < RATE_TTL:
        return _rate_cache["rate"]

    # 1. exchangerate.host live endpoint
    try:
        if FX_ACCESS_KEY:
            r = requests.get(
                "https://api.exchangerate.host/live",
                params={"access_key": FX_ACCESS_KEY, "base": "USD", "symbols": "MXN"},
                timeout=8
            )
            r.raise_for_status()
            j = r.json()
            if j.get("success") and "USDMXN" in j.get("quotes", {}):
                rate = float(j["quotes"]["USDMXN"])
                _rate_cache.update({"rate": rate, "ts": now})
                logging.info("FX rate fetched (exchangerate.host): 1 USD = %.4f MXN", rate)
                return rate
    except Exception as e:
        logging.debug("exchangerate.host attempt failed: %s", e)

    # 2. Open Provider Fallback
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=6)
        r.raise_for_status()
        j = r.json()
        if "rates" in j and "MXN" in j["rates"]:
            rate = float(j["rates"]["MXN"])
            _rate_cache.update({"rate": rate, "ts": now})
            logging.info("FX rate fetched (open API): 1 USD = %.4f MXN", rate)
            return rate
    except Exception as e:
        logging.debug("Open FX provider failed: %s", e)

    logging.warning("Using fallback FX rate: %.4f", USD_FX_FALLBACK)
    return USD_FX_FALLBACK


USD_to_MXN = get_usd_mxn_rate()
MXN_to_USD = 1.0 / USD_to_MXN


# --- Excel Reference Table Loading ---
def _load_tables():
    try:
        labor_df = pd.read_excel(LABOR_XLSX)
        parts_df = pd.read_excel(PARTS_XLSX)
    except Exception as e:
        logging.error("Failed to load Excel reference files: %s", e)
        # Create dummy DFs to prevent crash if files missing during testing
        labor_df = pd.DataFrame(columns=["Reclassified_Type", "Car_Model_Name", "Car_Year", "Labor (MXN per hr)"])
        parts_df = pd.DataFrame(columns=["Reclassified_Type", "Car_Model_Name", "Car_Year", "Amount_MXN"])

    def normalize_label(s):
        return str(s).lower().replace(" ", "").replace("-", "")

    if not labor_df.empty:
        labor_df["Reclassified_Type_norm"] = labor_df["Reclassified_Type"].apply(normalize_label)
    if not parts_df.empty:
        parts_df["Reclassified_Type_norm"] = parts_df["Reclassified_Type"].apply(normalize_label)

    return labor_df, parts_df


labor_df, parts_df = _load_tables()


# --- Local Lookup Logic ---
# --- 1. UPDATED LOCAL LOOKUP (With Severity Logic) ---
# --- UPDATED LOCAL LOOKUP (Matches your CSV data exactly) ---
def lookup_local_cost(label: str, car_model: str, car_year: int, severity: float) -> Optional[Dict[str, Any]]:
    label_norm = str(label).lower().replace(" ", "").replace("-", "")

    # Map common YOLO labels to your Excel spelling if needed
    search_label = label_norm
    if "quarter" in label_norm or "quater" in label_norm:
        search_label = "quarterpaneldent"
    elif "running" in label_norm:
        search_label = "runningboarddent"
    elif "windscreen" in label_norm or "windshield" in label_norm:
        search_label = "frontwindshielddamage"

    # Filter Data - Note: We filter by Model Name but average across years if specific year missing
    # This makes it robust for 2024 vs 2022 mismatches
    ref_parts = parts_df[
        (parts_df["Reclassified_Type_norm"] == search_label) &
        (parts_df["Car_Model_Name"].str.contains(car_model, case=False, na=False))
    ]
    ref_labor = labor_df[
        (labor_df["Reclassified_Type_norm"] == search_label) &
        (labor_df["Car_Model_Name"].str.contains(car_model, case=False, na=False))
    ]

    if not ref_parts.empty and not ref_labor.empty:
        # Get Average Costs
        base_part_cost = float(ref_parts["Amount_MXN"].mean())
        base_labor_rate = float(ref_labor["Labor (MXN per hr)"].mean())

        # SCALING LOGIC
        scaling_factor = 0.5 + max(0.1, min(severity, 0.9))

        adjusted_part_cost = base_part_cost * scaling_factor
        adjusted_hours = 2.0 * scaling_factor
        labor_total = adjusted_hours * base_labor_rate
        total = adjusted_part_cost + labor_total

        return {
            "parts_cost": adjusted_part_cost,
            "labor_rate": base_labor_rate,
            "hours": adjusted_hours,
            "labor_cost": labor_total,
            "total": total,
            "source": "local_excel_backup"
        }
    return None

# --- UPDATED Estimator API Integration (Connects to Flask) ---
# --- 1. UPDATED API CLIENT (Captures Notes) ---
def call_estimator_api(label, severity, car_model, car_year, region="Mexico", currency="MXN"):
    payload = {
        "label": label,
        "severity": float(severity),
        "car_model": car_model,
        "car_year": int(car_year),
        "region": region,
        "currency": currency
    }
    headers = {"Content-Type": "application/json"}

    # Retry logic for robustness
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("http://", HTTPAdapter(max_retries=retries))

    logging.debug("Calling Flask Estimator: %s | Payload: %s", ESTIMATOR_URL, payload)

    try:
        resp = session.post(ESTIMATOR_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        full_response = resp.json()

        if "estimate" not in full_response:
            logging.error("API response missing 'estimate' key: %s", full_response)
            return {"total": 0.0, "source": "api_error", "notes": "API Error: Invalid Format"}

        est = full_response["estimate"]

        return {
            "total": float(est.get("total_cost", 0.0)),
            "parts_cost": float(est.get("parts_cost", 0.0)),
            "labor_cost": float(est.get("labor_cost", 0.0)),
            "hours": float(est.get("hours_of_labor", 0.0)),
            "notes": est.get("notes", "No notes provided by API"),  # <--- Capture Notes
            "source": "api_flask"
        }

    except requests.RequestException as exc:
        logging.warning("Estimator API connection failed: %s", exc)
        raise

# --- Main Logic Controller (Simplified) ---
def estimate_damage_cost_mxn(label: str, box_area: float, img_area: float) -> Tuple[float, Dict[str, Any]]:
    """
    Main controller: Tier 1 (Flask API) -> Tier 2 (Local Excel) -> Tier 3 (Fallback)
    """
    # Calculate Severity
    damage_fraction = box_area / img_area
    damage_fraction = max(0.01, min(damage_fraction, 0.5))
    severity = damage_fraction / 0.5

    # --- TIER 1: Primary Flask API ---
    try:
        # Note: We removed labor_rate_mxn from arguments as Flask determines it or uses defaults
        api_data = call_estimator_api(label, severity, CAR_MODEL, CAR_YEAR)
        return severity, api_data
    except Exception as e:
        logging.warning(f"Tier 1 (Flask API) failed: {e}. Moving to Tier 2 (Local).")

    # --- TIER 2: Local Excel Data (Backup) ---
    try:
        logging.info("Attempting Tier 2: Local Excel Backup")
        local_data = lookup_local_cost(label, CAR_MODEL, CAR_YEAR, severity)

        if local_data:
            return severity, {
                "total": local_data["total"],
                "parts_cost": local_data["parts_cost"],
                "labor_cost": local_data["labor_cost"],
                "hours": local_data["hours"],
                "source": "local_backup"
            }
    except Exception as e:
        logging.error(f"Tier 2 (Local) failed: {e}")

    # --- TIER 3: Ultimate Safety Net ---
    logging.error("All estimators failed. Using Hardcoded Fallback.")
    return severity, {
        "total": 1500.0,
        "parts_cost": 875.0,
        "labor_cost": 625.0,
        "hours": 2.5,
        "source": "hard_fallback"
    }
# --- Verification Logic ---
def verify_estimate(api_data, local_data, tolerance=0.15):
    if not local_data:
        return "provisional", api_data

    api_total = api_data.get("total", 0.0)
    local_total = local_data.get("total", 0.0)

    if local_total == 0: return "provisional", api_data

    diff = abs(api_total - local_total) / local_total

    if diff <= tolerance:
        return "accepted", api_data
    elif diff <= 2 * tolerance:
        # Blend: 50% local, 50% api
        blended = (0.5 * local_total) + (0.5 * api_total)
        api_data["total"] = blended
        api_data["note"] = "Blended with local data"
        return "blended", api_data
    else:
        return "flagged", {"api": api_data, "local": local_data}


# --- Main Execution Pipeline ---
# --- 3. UPDATED PIPELINE (Prints Notes) ---
def run_pipeline(image_path_in):
    if YOLO is None:
        logging.error("Cannot run pipeline without YOLO.")
        return

    # Check for empty Excel files
    if labor_df.empty or parts_df.empty:
        logging.error("!!! WARNING: Excel Reference Tables are EMPTY. Results may be provisional. !!!")

    img = cv2.imread(image_path_in)
    if img is None:
        logging.error("Could not load image: %s", image_path_in)
        return

    h, w, _ = img.shape
    img_area = h * w

    # Get visual fallback rate
    try:
        avg_labor = float(labor_df["Labor (MXN per hr)"].mean())
    except:
        avg_labor = 250.0

    model = YOLO(YOLO_WEIGHTS)
    results = model(img, verbose=False)
    class_labels = model.names

    total_claim_mxn = 0.0
    line_items = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box, h_box = x2 - x1, y2 - y1
            box_area = w_box * h_box
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = class_labels[cls]

            if conf > 0.3:
                # A. Estimation
                severity, est_data = estimate_damage_cost_mxn(label, box_area, img_area)
                local_data = lookup_local_cost(label, CAR_MODEL, CAR_YEAR, severity)
                status, final_data = verify_estimate(est_data, local_data)

                # B. Data Extraction
                if status == "flagged":
                    data_src = final_data["local"] if final_data["local"] else final_data["api"]
                else:
                    data_src = final_data
                    total_claim_mxn += data_src.get("total", 0.0)

                # C. Recover Rate
                labor_cost = data_src.get("labor_cost", 0.0)
                hours = data_src.get("hours", 0.0)
                if data_src.get("labor_rate"):
                    final_rate = data_src.get("labor_rate")
                elif hours > 0:
                    final_rate = labor_cost / hours
                else:
                    final_rate = avg_labor

                # D. Save Item (Including Notes!)
                line_items.append({
                    "label": label,
                    "severity": severity,
                    "status": status,
                    "rate": final_rate,
                    "hours": hours,
                    "labor": labor_cost,
                    "parts": data_src.get("parts_cost", 0.0),
                    "total": data_src.get("total", 0.0),
                    "notes": est_data.get("notes", "No notes")  # Use est_data for original AI notes
                })

                # E. Draw UI
                color = (0, 255, 0) if status == "accepted" else (0, 165, 255)
                if status == "flagged": color = (0, 0, 255)
                item_total = data_src.get("total", 0.0)
                if cvzone:
                    cvzone.cornerRect(img, (x1, y1, w_box, h_box), t=2, colorR=color)
                    text = f"{label} | ${item_total:,.0f}"
                    cvzone.putTextRect(img, text, (x1, y1 - 10), scale=0.7, thickness=1, colorR=color)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{label} ${item_total:.0f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                                2)

    # 5. Print Final Report
    print("\n" + "=" * 105)
    print(f" VEHICLE DAMAGE REPORT: {CAR_MODEL} ({CAR_YEAR})")
    print("=" * 105)
    print(
        f"{'DAMAGE TYPE':<25} {'SEV':<5} {'STATUS':<12} {'RATE':<8} {'HOURS':<6} {'LABOR':>12} {'PARTS':>12} {'TOTAL':>12}")
    print("-" * 105)

    for item in line_items:
        print(
            f"{item['label']:<25} {item['severity']:<5.2f} {item['status']:<12} ${item['rate']:<7.0f} {item['hours']:<6.1f} ${item['labor']:>11,.0f} ${item['parts']:>11,.0f} ${item['total']:>11,.0f}")

    print("-" * 105)
    print(f"GRAND TOTAL ESTIMATE: {'$ {:,.2f} MXN'.format(total_claim_mxn):>96}")
    print("=" * 105)

    # 6. Print Notes Section
    if line_items:
        print("\n--- DETAILED NOTES ---")
        from textwrap import fill
        for item in line_items:
            print(f"[{item['label']}] (Sev: {item['severity']:.2f}):")
            print(fill(item['notes'], width=100, initial_indent="  ", subsequent_indent="  "))
            print("")
    print("=" * 105 + "\n")

    cv2.imshow("Estimation", img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure a file exists or use the default variable
    target_img = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    run_pipeline(target_img)