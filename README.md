# Damage-API
### Damage Estimator Pipeline README

#### Overview
A single-file image → cost estimation pipeline that converts object detections into provisional repair claim amounts. The pipeline combines **YOLO detection**, an **external estimator API** (primary), **Google Generative Language** (fallback), and **local Excel reference lookups** for verification and blending. Results are aggregated into **accepted** (auto-included) and **flagged** (doubtful) buckets with clear provenance for audit.

---

#### Architecture
- **Input**: image file (single image mode).  
- **Detection**: YOLO model returns boxes, class ids, confidences.  
- **Estimation sources**:
  - **Primary**: external estimator API (`ESTIMATOR_URL` + Bearer token).  
  - **Fallback 1**: Google Generative Language (text-bison) via `GOOGLE_API_KEY`.  
  - **Fallback 2**: local Excel lookups (`LABOR_XLSX`, `PARTS_XLSX`) or default heuristic.  
- **Verification**: compare API vs local totals and decide `accepted`, `blended`, `provisional`, or `flagged`.  
- **Currency**: USD↔MXN conversion via cached FX lookup with multi-provider fallback.  
- **Output**: annotated image, concise logs per detection, and summary totals.

---

#### Per-detection Workflow
1. **Detect**
   - Run YOLO and iterate `for r in results: for box in r.boxes:`.
   - Compute `box_area = w * h`, `img_area = img_h * img_w`.
   - Skip low confidence boxes (`conf <= 0.3`).

2. **Measure**
   - Compute `damage_fraction = clamp(box_area / img_area, 0.01, 0.5)`.
   - Compute `severity = damage_fraction / 0.5`.

3. **Estimate**
   - Call primary estimator:
     ```python
     api_resp = call_estimator_api(label, severity, CAR_MODEL, CAR_YEAR, damage_fraction)
     api_resp["source"] = "api"
     ```
   - If API fails, call Google fallback:
     ```python
     severity, parsed = estimate_with_google(label, box_area, img_area, labor_rate_mxn)
     parsed["source"] = "google"
     ```
   - If Google fails, use local lookup:
     ```python
     local = lookup_local_cost(label, CAR_MODEL, CAR_YEAR)
     # compute iva and total if local found, else default parts+labor
     ```

4. **Verify and Blend**
   - Run `verify_estimate(api_data, local_data, tolerance=0.15)` to decide status and produce `verified_data`.
   - Extract total robustly:
     ```python
     est_total_mxn, total_source = _extract_total_from_verified(verified_data)
     ```

5. **Accumulate and Annotate**
   - If `status` in `["accepted","blended","provisional"]` add to `total_claim_mxn`.
   - If `status == "flagged"` add to `flagged_items` and `flagged_total_mxn`.
   - Log one concise `INFO` per detection and overlay text on image.

---

#### Verification and Blending Code
**Policy**  
- **accepted**: difference ≤ `tolerance` (default 15%).  
- **blended**: `tolerance < diff ≤ 2*tolerance` → weighted average (70% local, 30% API).  
- **flagged**: `diff > 2*tolerance` → keep both for manual review.  
- **provisional**: no local reference available.

**Implementation**
```python
def verify_estimate(api_data, local_data, tolerance=0.15):
    if local_data is None:
        return "provisional", api_data
    api_total = float(api_data["total"])
    local_total = float(local_data["total"])
    diff = abs(api_total - local_total) / local_total
    if diff <= tolerance:
        api_data["source"] = api_data.get("source", "api")
        return "accepted", api_data
    elif diff <= 2 * tolerance:
        blended_total = 0.7 * local_total + 0.3 * api_total
        api_data["total"] = blended_total
        api_data["source"] = "blended"
        return "blended", api_data
    else:
        return "flagged", {"api": api_data, "local": local_data}
```

**Notes**
- Keep `source` tags (`api`, `google`, `local`, `blended`) for audit.  
- Store `flagged_items` with `api_total`, `local_total`, `conf`, and `chosen_source`.

---

#### Fallbacks FX and Currency
**FX lookup strategy**
- Cached function `get_usd_mxn_rate()` with TTL (`RATE_TTL` seconds).  
- Provider order:
  1. Open provider `open.er-api.com` (no key).  
  2. apilayer endpoint with header `apikey` (if `FX_ACCESS_KEY` present).  
  3. exchangerate.host with `access_key` query param.  
  4. Static fallback `USD_FX_FALLBACK` if all fail.  
- Cache result in `_rate_cache` to avoid repeated network calls.

**IVA and totals**
- Compute `iva = subtotal * IVA_RATE` and `total = subtotal + iva`.  
- Convert MXN→USD for display: `MXN_to_USD = 1.0 / USD_to_MXN`.

**Robustness**
- Use `requests.Session()` with `Retry` for estimator calls.  
- Short timeouts for FX calls and fail-fast behavior to trigger fallbacks.

---

#### Quickstart Configuration and Logging
**Environment variables**
```
ESTIMATOR_URL=http://127.0.0.1:8080
ESTIMATOR_API_KEY=...
GOOGLE_API_KEY=...
FX_ACCESS_KEY=...
YOLO_WEIGHTS=Weights/best.pt
LABOR_XLSX=FINAL_LaborRate_...xlsx
PARTS_XLSX=FINAL_PartCost_...xlsx
CAR_MODEL=Nissan Pathfinder
CAR_YEAR=2012
USD_FX_FALLBACK=18.5
IVA_RATE=0.16
```

**Logging recommendations**
- Convert `print()` to `logging.debug()` and keep only essential `INFO` lines:
  - Startup summary, `FX rate fetched`, `Using labor rate`, one concise per-detection `INFO`, and final summary lines.
- Suppress noisy library logs:
```python
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
```
- Optional `LOG_LEVEL` env toggle for temporary `DEBUG` runs.

**Run**
- Ensure `.env` is present and `load_dotenv()` runs before reading env vars.  
- Start script from the same terminal used to verify env values to avoid IDE env caching issues.

---

#### Testing and Extensions
- **Unit tests**: test `verify_estimate()` with synthetic totals; mock `call_estimator_api()` to simulate success/failure.  
- **Integration tests**: stub estimator and FX providers to validate fallback behavior and final aggregation.  
- **Extensions**:
  - Export `flagged_items` CSV for manual review.  
  - Add a UI to accept/reject flagged items and recalculate totals.  
  - Add rate-limiting and batching for high-throughput image queues.

---

#### Summary
This pipeline turns detections into auditable monetary estimates by combining external models, generative fallback, and local references with conservative verification rules. The code is organized for reliability, traceability, and easy tuning of tolerance, provenance, and logging.
