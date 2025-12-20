# llm_estimator.py — Gemini Flash via v1beta generateContent (ROBUST)

import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Just print a warning instead of crashing app on start
    print("WARNING: GEMINI_API_KEY not set.")

DEFAULT_REGION = os.getenv("REGION", "Mexico")
GEMINI_MODEL_NAME = "gemini-flash-latest"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL_NAME}:generateContent"
)

app = Flask(__name__)

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are an experienced auto body repair estimator.

Given:
- damaged car part (string),
- severity (0.0 - 1.0),
- car model, year, and region,

Return a STRICT JSON object with this shape:

{
  "parts_cost_mxn": 0,
  "labor_cost_mxn": 0,
  "total_cost_mxn": 0,
  "parts_cost_usd": 0,
  "labor_cost_usd": 0,
  "total_cost_usd": 0,
  "labor_rate_mxn": 0,
  "hours_of_labor": 0,
  "exchange_rate_used": 0,
  "notes": "short explanation"
}

Rules:
- Calculate MXN costs first based on the local market (Mexico).
- Use a realistic, current exchange rate for USD conversion.
- Return the rate you used in "exchange_rate_used".
- total_cost = parts_cost + labor_cost.
- Do NOT include any extra keys or text.
"""


def call_llm_estimator(label, severity, car_model, car_year, region):
    prompt = f"""
{SYSTEM_PROMPT}

Estimate repair costs for:
- Part: {label}
- Severity: {severity}
- Car: {car_model} ({car_year})
- Region: {region}

Return ONLY the JSON object.
"""
    params = {"key": GEMINI_API_KEY}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        resp = requests.post(GEMINI_ENDPOINT, params=params, json=payload, timeout=60)
        resp.raise_for_status()  # Raise error for 4xx/5xx

        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Clean Markdown
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        # Basic Validation
        if "total_cost_mxn" not in result:
            raise ValueError("Missing 'total_cost_mxn' in response")

        return result

    except Exception as e:
        # Return a "Safe Fallback" instead of crashing
        print(f"LLM Error: {e}")
        return {
            "parts_cost_mxn": 0, "labor_cost_mxn": 0, "total_cost_mxn": 0,
            "parts_cost_usd": 0, "labor_cost_usd": 0, "total_cost_usd": 0,
            "labor_rate_mxn": 0, "hours_of_labor": 0, "exchange_rate_used": 20.0,
            "notes": f"API Error: {str(e)}"
        }


@app.route("/estimate", methods=["POST"])
def estimate():
    # 1. Safe JSON Parsing
    try:
        body = request.get_json(force=True)
        if not body: body = {}
    except:
        body = {}

    # 2. Extract Fields (With Safe Defaults)
    label = body.get("label")
    if not label:
        return jsonify({"error": "Missing required field 'label'"}), 400

    # Safe Severity
    try:
        raw_sev = body.get("severity")
        severity = float(raw_sev) if raw_sev is not None else 0.5
        severity = max(0.0, min(severity, 1.0))
    except:
        severity = 0.5

    # Safe Car Model/Year
    car_model = str(body.get("car_model", "Unknown Model"))
    try:
        car_year = int(body.get("car_year", 2015))
    except:
        car_year = 2015

    region = str(body.get("region", DEFAULT_REGION))

    # 3. Call LLM (Safe Wrapper)
    est = call_llm_estimator(label, severity, car_model, car_year, region)

    # 4. Return Full Data (Client needs these keys!)
    return jsonify({
        "label": label,
        "severity": severity,  # <--- Critical for client
        "car_model": car_model,
        "car_year": car_year,
        "region": region,
        "estimate": est
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)