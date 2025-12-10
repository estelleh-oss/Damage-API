import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

# Try v1beta first
url = "https://generativelanguage.googleapis.com/v1beta/models"
params = {"key": API_KEY}

print(f"Requesting models from: {url}")
resp = requests.get(url, params=params, timeout=30)
print("Status code:", resp.status_code)
print("Raw response:")
print(resp.text)