# llm_estimator.py — Gemini Flash via v1beta generateContent

import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# ------------------------------
# 1. Load env + API key
# ------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not set. Put it in .env as GEMINI_API_KEY=AIza..."
    )

DEFAULT_CURRENCY = os.getenv("CURRENCY", "MXN")
DEFAULT_REGION = os.getenv("REGION", "Mexico")

# Use a model we KNOW exists from list_models:
#   "name": "models/gemini-flash-latest"
GEMINI_MODEL_NAME = "gemini-flash-latest"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL_NAME}:generateContent"
)

app = Flask(__name__)

SYSTEM_PROMPT = """
You are an experienced auto body repair estimator for insurance claims.

Given:
- damaged car part (string, like "rear-bumper-dent"),
- severity between 0 and 1 (0.1 = light, 1.0 = very severe),
- car model, year, and region,

you must return a STRICT JSON object with exactly this shape:

{
  "currency": "MXN",
  "parts_cost": 0,
  "labor_cost": 0,
  "total_cost": 0,
  "hours_of_labor": 0,
  "notes": "short explanation"
}

Rules:
- Use the requested currency (default MXN).
- All cost fields are numbers (float), no currency symbols.
- total_cost = parts_cost + labor_cost.
- hours_of_labor is a realistic number.
- notes: <= 3 sentences, explain assumptions & uncertainty.
- Do NOT include any extra keys or text, ONLY the JSON object.
"""


def call_llm_estimator(label: str,
                       severity: float,
                       car_model: str,
                       car_year: int,
                       region: str,
                       currency: str) -> dict:
    """
    Call Gemini Flash and get a JSON cost estimate.
    """

    prompt = f"""
{SYSTEM_PROMPT}

Now, estimate repair costs for the following car damage:

- Damaged part label: {label}
- Severity (0 to 1): {severity}
- Car model: {car_model}
- Car year: {car_year}
- Region: {region}
- Desired currency: {currency}

Return ONLY a JSON object with the keys:
["currency","parts_cost","labor_cost","total_cost","hours_of_labor","notes"].
No extra text.
"""

    params = {"key": GEMINI_API_KEY}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    resp = requests.post(GEMINI_ENDPOINT, params=params, json=payload, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

    data = resp.json()

    # Extract the text response from candidates
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response format: {data}") from e

    # Try to parse JSON from the text
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()
        result = json.loads(cleaned)

    required_keys = [
        "currency",
        "parts_cost",
        "labor_cost",
        "total_cost",
        "hours_of_labor",
        "notes",
    ]
    for k in required_keys:
        if k not in result:
            raise ValueError(f"Missing key '{k}' in LLM response: {result}")

    return result


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/estimate", methods=["POST"])
def estimate():
    """
    Example request JSON:
    {
      "label": "rear-bumper-dent",
      "severity": 0.8,
      "car_model": "Nissan Pathfinder",
      "car_year": 2012,
      "region": "Mexico",
      "currency": "MXN"
    }
    """
    body = request.get_json(force=True) or {}

    label = body.get("label")
    if not label:
        return jsonify({"error": "Missing required field 'label'"}), 400

    severity = float(body.get("severity", 0.5))
    car_model = body.get("car_model", "Unknown model")
    car_year = int(body.get("car_year", 2015))
    region = body.get("region", DEFAULT_REGION)
    currency = body.get("currency", DEFAULT_CURRENCY)

    try:
        est = call_llm_estimator(
            label=label,
            severity=severity,
            car_model=car_model,
            car_year=car_year,
            region=region,
            currency=currency,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "label": label,
        "severity": severity,
        "car_model": car_model,
        "car_year": car_year,
        "region": region,
        "estimate": est
    })


if __name__ == "__main__":
    print("Starting Gemini Flash estimator API on http://0.0.0.0:8080 ...")
    app.run(host="0.0.0.0", port=8080, debug=True)