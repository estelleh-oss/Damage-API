#!/usr/bin/env python3
"""
Final Manual Verification Client (Exact Match Version)
- Input: Assumes user types EXACT Excel labels (e.g. "front-bumper-dent").
- Logic: Linear Interpolation + Generic Fallback for missing models.
- Rate: Dynamic from Gemini.
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 1. CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("requests").setLevel(logging.WARNING)

ESTIMATOR_URL = os.getenv("ESTIMATOR_URL", "http://127.0.0.1:8080/estimate")
LABOR_XLSX = os.getenv("LABOR_XLSX", "FINAL_LaborRate_with_Reclassification_MXN_Thresholds_and_CarSplit.xlsx")
PARTS_XLSX = os.getenv("PARTS_XLSX", "FINAL_PartCost_with_Reclassification_MXN_Thresholds_and_CarSplit.xlsx")


# --- 2. LOAD EXCEL ---
def _load_tables():
    try:
        labor_df = pd.read_excel(LABOR_XLSX)
        parts_df = pd.read_excel(PARTS_XLSX)
    except:
        return pd.DataFrame(), pd.DataFrame()

    # Normalization helper to ensure strict matching ignores spaces/dashes
    def normalize_label(s):
        return str(s).lower().replace(" ", "").replace("-", "")

    if not labor_df.empty:
        labor_df["Reclassified_Type_norm"] = labor_df["Reclassified_Type"].apply(normalize_label)
    if not parts_df.empty:
        parts_df["Reclassified_Type_norm"] = parts_df["Reclassified_Type"].apply(normalize_label)

    return labor_df, parts_df


labor_df, parts_df = _load_tables()


# --- 3. HELPER: INTERPOLATION ---
def interpolate_cost(target_severity, points):
    if not points: return 0.0
    points.sort(key=lambda x: x[0])

    # Case A: Single Point Extrapolation
    if len(points) == 1:
        known_sev, known_cost = points[0]
        if known_sev == 0: return known_cost
        ratio = target_severity / known_sev
        ratio = max(0.5, min(ratio, 2.5))
        return known_cost * ratio

    # Case B: Multi-Point Interpolation
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    if target_severity < xs[0]:
        ratio = target_severity / xs[0]
        return ys[0] * max(0.5, ratio)
    if target_severity > xs[-1]:
        ratio = target_severity / xs[-1]
        return ys[-1] * min(1.5, ratio)

    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        y1, y2 = ys[i], ys[i + 1]
        if x1 <= target_severity <= x2:
            gap = x2 - x1
            if gap == 0: return y1
            ratio = (target_severity - x1) / gap
            return y1 + ((y2 - y1) * ratio)
    return ys[-1]


# --- 4. API CLIENT ---
def call_api(label, severity, car_model, car_year):
    payload = {
        "label": label, "severity": float(severity),
        "car_model": car_model, "car_year": int(car_year),
        "region": "Mexico"
    }
    try:
        session = requests.Session()
        session.mount("http://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))
        resp = session.post(ESTIMATOR_URL, json=payload, timeout=30)
        data = resp.json()

        if "estimate" not in data:
            return {"total": 0.0, "source": "error", "rate": 20.0}

        est = data["estimate"]
        gemini_rate = float(est.get("exchange_rate_used", 20.0))

        return {
            "total": float(est.get("total_cost_mxn", 0.0)),
            "parts_cost": float(est.get("parts_cost_mxn", 0.0)),
            "labor_cost": float(est.get("labor_cost_mxn", 0.0)),
            "hours": float(est.get("hours_of_labor", 0.0)),
            "labor_rate": float(est.get("labor_rate_mxn", 0.0)),
            "rate": gemini_rate,
            "notes": est.get("notes", ""),
            "source": "api"
        }
    except Exception as e:
        return {"total": 0.0, "source": "fail", "notes": str(e), "rate": 20.0}


# --- 5. LOCAL LOOKUP (EXACT MATCH + FALLBACK) ---
def lookup_local(label, car_model, severity):
    # Normalize input to ensure matches against Excel (e.g. "Front-Bumper" == "frontbumper")
    label_norm = str(label).lower().replace(" ", "").replace("-", "")

    # 1. ATTEMPT 1: Strict Match (Label + Car Model)
    ref_parts = parts_df[
        (parts_df["Reclassified_Type_norm"] == label_norm) &
        (parts_df["Car_Model_Name"].str.contains(car_model, case=False, na=False))
        ]
    ref_labor = labor_df[
        (labor_df["Reclassified_Type_norm"] == label_norm) &
        (labor_df["Car_Model_Name"].str.contains(car_model, case=False, na=False))
        ]

    # 2. ATTEMPT 2: Generic Fallback (Label Only)
    # Necessary because "Pathfinder" might not have Bumper data, but "Leaf" does.
    if ref_parts.empty or ref_labor.empty:
        ref_parts = parts_df[parts_df["Reclassified_Type_norm"] == label_norm]
        ref_labor = labor_df[labor_df["Reclassified_Type_norm"] == label_norm]

    if ref_parts.empty or ref_labor.empty:
        return None

    # 3. Prepare Data
    part_points = ref_parts.groupby("Damage_Severity_Threshold")["Amount_MXN"].mean().reset_index().values.tolist()

    temp_labor = ref_labor.copy()
    temp_labor["valid_duration"] = temp_labor["Estimated Total Duration (If Applicable)"].fillna(2.0)
    temp_labor["row_cost"] = temp_labor["valid_duration"] * temp_labor["Labor (MXN per hr)"]

    labor_cost_points = temp_labor.groupby("Damage_Severity_Threshold")["row_cost"].mean().reset_index().values.tolist()
    duration_points = temp_labor.groupby("Damage_Severity_Threshold")[
        "valid_duration"].mean().reset_index().values.tolist()

    # 4. Interpolate
    final_part_cost = interpolate_cost(severity, part_points)
    final_labor_cost = interpolate_cost(severity, labor_cost_points)
    final_hours = interpolate_cost(severity, duration_points)

    total_mxn = final_part_cost + final_labor_cost
    implied_rate = final_labor_cost / final_hours if final_hours > 0 else 0

    return {
        "total": total_mxn, "parts": final_part_cost,
        "labor": final_labor_cost, "hours": final_hours, "rate": implied_rate
    }


# --- 6. VERIFICATION (ZERO CHECK INCLUDED) ---
def verify(api_data, local_data):
    if not local_data: return "PROVISIONAL", api_data

    api_total = api_data.get("total", 0.0)
    local_total = local_data.get("total", 0.0)

    # SAFETY: If API crashed (0), force Local
    if api_total == 0 or api_total is None:
        fallback = api_data.copy()
        fallback.update({
            "total": local_total,
            "parts_cost": local_data["parts"],
            "labor_cost": local_data["labor"],
            "hours": local_data["hours"],
            "labor_rate": local_data["rate"],
            "notes": "[ERROR] API Failed ($0). Using Local Anchor."
        })
        return "LOCAL_FB", fallback

    if local_total == 0: return "PROVISIONAL", api_data

    diff = abs(api_total - local_total) / local_total

    # Relaxed Thresholds for Demo
    if diff <= 0.15:
        return "VERIFIED", api_data
    else:
        api_data["notes"] = f"[FLAGGED] High Variance ({diff:.1%}) vs Local Anchor (${local_total:,.0f})."
        return "FLAGGED", api_data


# --- 7. MAIN EXECUTION ---
def main():
    print("\n" + "=" * 60)
    print("   VEHICLE DAMAGE ESTIMATOR (EXACT INPUT MODE)")
    print("=" * 60)

    print("--- VEHICLE DETAILS ---")
    def_model = "Nissan Pathfinder"
    def_year = "2012"
    in_model = input(f"Car Model [{def_model}]: ").strip() or def_model
    in_year = input(f"Car Year  [{def_year}]: ").strip() or def_year
    try:
        CAR_YEAR = int(in_year)
    except:
        CAR_YEAR = 2012

    cart_items = []

    print(f"\n--- ADD DAMAGE ITEMS for {in_model} ---")
    while True:
        print(f"\n[Item #{len(cart_items) + 1}]")
        label = input("Damage Label: ").strip()
        if label.lower() in ["stop", "done", "exit", "quit", ""]: break

        sev_str = input("Severity (0.0 - 1.0): ").strip()
        try:
            severity = float(sev_str)
        except:
            continue

        print(" > Estimating...")

        api_res = call_api(label, severity, in_model, CAR_YEAR)
        loc_res = lookup_local(label, in_model, severity)
        status, final = verify(api_res, loc_res)

        rate_used = api_res.get("rate", 20.0)
        final["label"] = label
        final["severity"] = severity
        final["status"] = status
        final["rate_used"] = rate_used

        cart_items.append(final)
        print(f" > Added: {label} (${final.get('total', 0):,.0f} MXN)")

    if not cart_items: return

    print("\n\n" + "=" * 120)
    print(f" DAMAGE ESTIMATION REPORT: {in_model} ({CAR_YEAR})")
    print("=" * 120)
    print(
        f"{'DAMAGE TYPE':<25} {'SEV':<5} {'STATUS':<10} {'RATE (MXN)':<12} {'HRS':<5} {'LABOR':>10} {'PARTS':>10} {'MXN TOTAL':>12} {'USD TOTAL':>12}")
    print("-" * 120)

    grand_total_mxn = 0.0
    grand_total_usd = 0.0

    for item in cart_items:
        total_mxn = item.get('total', 0)
        fx_rate = item.get('rate_used', 20.0)
        total_usd = total_mxn / fx_rate if fx_rate else 0

        grand_total_mxn += total_mxn
        grand_total_usd += total_usd

        if item.get("labor_rate", 0) > 0:
            rate = item["labor_rate"]
        elif item.get("hours", 0) > 0:
            rate = item["labor_cost"] / item["hours"]
        else:
            rate = 0.0

        print(
            f"{item['label'][:24]:<25} {item['severity']:<5.2f} {item['status']:<10} ${rate:<11,.0f} {item.get('hours', 0):<5.1f} ${item.get('labor_cost', 0):>9,.0f} ${item.get('parts_cost', 0):>9,.0f} ${total_mxn:>11,.0f} ${total_usd:>11.2f}")

    print("-" * 120)
    print(f"GRAND TOTAL: {'$ {:,.2f} MXN'.format(grand_total_mxn):>99} {'$ {:,.2f} USD'.format(grand_total_usd):>13}")
    print("=" * 120)

    print("\n--- NOTES ---")
    for i, item in enumerate(cart_items):
        if item.get("notes"):
            print(f"{i + 1}. {item['label']} ({item['status']}): {item['notes']}")
    print("\n")


if __name__ == "__main__":
    main()