from flask import Flask, request, jsonify
import os
import pandas as pd

# load same Excel files as your main script
LABOR_XLSX = os.getenv("LABOR_XLSX", "FINAL_LaborRate_with_Reclassification_MXN_Thresholds_and_CarSplit.xlsx")
PARTS_XLSX = os.getenv("PARTS_XLSX", "FINAL_PartCost_with_Reclassification_MXN_Thresholds_and_CarSplit.xlsx")
IVA_RATE = float(os.getenv("IVA_RATE", "0.16"))

labor_df = pd.read_excel(LABOR_XLSX)
parts_df = pd.read_excel(PARTS_XLSX)

def normalize_label(s):
    return str(s).lower().replace(" ", "").replace("-", "")

labor_df["Reclassified_Type_norm"] = labor_df["Reclassified_Type"].apply(normalize_label)
parts_df["Reclassified_Type_norm"] = parts_df["Reclassified_Type"].apply(normalize_label)

def lookup_local_cost(label, car_model, car_year):
    label_norm = normalize_label(label)
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
        labor_rate_usd = float(ref_labor["Labor (MXN per hr)"].mean())
        # use fallback FX or environment if needed
        usd_fx = float(os.getenv("USD_FX_FALLBACK", "18.5"))
        labor_rate_mxn = labor_rate_usd * usd_fx
        total = parts_cost + labor_rate_mxn
        return {"parts_cost": parts_cost, "labor_rate": labor_rate_mxn, "total": total}
    return None

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    j = request.get_json() or {}
    label = j.get("label")
    car_model = j.get("car_model", "Nissan Pathfinder")
    car_year = int(j.get("car_year", 2012))
    local = lookup_local_cost(label, car_model, car_year)
    if local:
        subtotal = local["total"]
        iva = subtotal * IVA_RATE
        total = subtotal + iva
        return jsonify({
            "label": label,
            "parts_cost": local["parts_cost"],
            "labor_hours": 1.0,
            "labor_cost": local["labor_rate"],
            "subtotal": subtotal,
            "iva": iva,
            "total": total
        })
    # fallback heuristic
    parts = 500.0
    labor_cost = 200.0
    subtotal = parts + labor_cost
    iva = subtotal * IVA_RATE
    total = subtotal + iva
    return jsonify({"label": label, "parts_cost": parts, "labor_hours": 1.0, "labor_cost": labor_cost,
                    "subtotal": subtotal, "iva": iva, "total": total})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
