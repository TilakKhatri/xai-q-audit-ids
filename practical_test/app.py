#!/usr/bin/env python3
"""
XAI-Q IDS Model Server  —  FastAPI
Upload a CSV file to /kdd/upload or /unsw/upload; get back predictions and
per-row resource metrics (latency, memory, CPU) for each sample.

Build & run (from project root):
  docker build -f practical_test/Dockerfile -t xai-ids-server .
  docker run -p 8080:8080 xai-ids-server

Test:
  curl http://localhost:8080/health
  curl -X POST http://localhost:8080/kdd/upload  -F "file=@kdd99.csv"
  curl -X POST http://localhost:8080/unsw/upload -F "file=@UNSW-NB15_3.csv"
"""

import csv
import io
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, UploadFile
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ---------------------------------------------------------------------------
# Logging — stdout + persistent file
# ---------------------------------------------------------------------------
LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "predict.log"),
    ],
)
log = logging.getLogger("ids")

# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))

MODELS: dict = {}
for _key, _fname in [
    ("kdd",  "rf_model_KDDCup99_10k.joblib"),
    ("unsw", "rf_model_UNSW-NB15_10k.joblib"),
]:
    _path = MODELS_DIR / _fname
    if not _path.exists():
        log.error("Model not found: %s", _path)
        continue
    _t = time.perf_counter()
    MODELS[_key] = joblib.load(_path)
    _m = MODELS[_key]
    log.info(
        "Loaded %-4s | n_features=%-3d | classes=%s | load_ms=%.0f",
        _key, _m.n_features_in_, _m.classes_.tolist(),
        (time.perf_counter() - _t) * 1000,
    )

_PROC = psutil.Process()

# ---------------------------------------------------------------------------
# CSV upload metadata for each dataset
# ---------------------------------------------------------------------------

# KDD99: CSV has a header row with quoted names; drop id + label cols
_KDD_DROP = {"id", "class", "predictedclass"}
_KDD_CAT  = {"protocol_type", "service", "flag"}

# UNSW-NB15: CSV has no header row; these are the column names in order
_UNSW_COLS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]
_UNSW_DROP = {"attack_cat", "label", "srcip", "dstip"}
_UNSW_CAT  = {"proto", "state", "service"}


def _parse_csv(text: str, fixed_cols: list | None, drop: set, cats: set
               ) -> tuple[np.ndarray, list]:
    """
    Parse a CSV string into a MinMaxScaled feature array.
    fixed_cols: provide column names when the CSV has no header (UNSW).
                Pass None to read names from the header row (KDD).
    Returns (X_scaled, row_ids).
    """
    if fixed_cols:
        rows  = [r for r in csv.reader(io.StringIO(text)) if len(r) == len(fixed_cols)]
        dicts = [dict(zip(fixed_cols, r)) for r in rows]
        cols  = [c for c in fixed_cols if c not in drop]
        ids   = [str(i) for i in range(len(dicts))]
    else:
        reader = csv.DictReader(io.StringIO(text))
        dicts  = list(reader)
        # strip surrounding quotes that KDD99 CSV wraps around column names
        cols   = [k.strip("' ") for k in reader.fieldnames if k.strip("' ") not in drop]
        ids    = [d.get("id", str(i)) for i, d in enumerate(dicts)]

    if not dicts:
        raise ValueError("No data rows found in CSV.")

    def _flt(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    # Build matrix: leave cat cols as strings temporarily
    X = []
    for d in dicts:
        # handle both plain and quoted key variants
        row = [str(d.get(c, d.get(f"'{c}'", ""))).strip() if c in cats
               else _flt(d.get(c, d.get(f"'{c}'", 0)))
               for c in cols]
        X.append(row)

    # LabelEncode every string column in-place
    for i, col in enumerate(cols):
        if col in cats:
            le  = LabelEncoder()
            enc = le.fit_transform([r[i] for r in X]).astype(float)
            for j, r in enumerate(X):
                r[i] = enc[j]

    X_np = np.array(X, dtype=np.float32)
    return MinMaxScaler().fit_transform(X_np), ids


# ---------------------------------------------------------------------------
# Prediction with resource measurement
# ---------------------------------------------------------------------------
def _run(model_key: str, features: list) -> dict:
    model = MODELS[model_key]
    X = np.array(features, dtype=np.float32).reshape(1, -1)

    if X.shape[1] != model.n_features_in_:
        raise ValueError(
            f"'{model_key}' expects {model.n_features_in_} features, got {X.shape[1]}"
        )

    mem_before = _PROC.memory_info().rss
    cpu_before = _PROC.cpu_times()
    t0         = time.perf_counter()

    pred  = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    lat_ms    = (time.perf_counter() - t0) * 1_000
    mem_after = _PROC.memory_info().rss
    cpu_after = _PROC.cpu_times()

    result = {
        "model":      model_key,
        "prediction": pred,
        "label":      "Attack" if pred == 1 else "Normal",
        "probabilities": {
            str(c): round(float(p), 4)
            for c, p in zip(model.classes_, proba)
        },
        "performance": {
            "latency_ms":      round(lat_ms, 3),
            "memory_rss_mb":   round(mem_after / 1_048_576, 2),
            "memory_delta_kb": round((mem_after - mem_before) / 1024, 2),
            "cpu_delta_s":     round(
                (cpu_after.user + cpu_after.system)
                - (cpu_before.user + cpu_before.system), 6
            ),
        },
    }
    log.info(
        "PREDICT model=%-4s label=%-6s latency_ms=%-8.3f rss_mb=%-6.1f cpu_s=%s",
        model_key, result["label"],
        result["performance"]["latency_ms"],
        result["performance"]["memory_rss_mb"],
        result["performance"]["cpu_delta_s"],
    )
    return result


def _batch(model_key: str, X: np.ndarray, ids: list) -> dict:
    if model_key not in MODELS:
        raise HTTPException(503, f"Model '{model_key}' not loaded.")
    results = []
    for rid, vec in zip(ids, X):
        r = _run(model_key, vec.tolist())
        r["row_id"] = rid
        log.info(
            "PREDICT model=%-4s label=%-6s latency_ms=%-8.3f rss_mb=%-6.1f",
            model_key, r["label"],
            r["performance"]["latency_ms"],
            r["performance"]["memory_rss_mb"],
        )
        results.append(r)
    attacks = sum(r["prediction"] for r in results)
    lats    = [r["performance"]["latency_ms"] for r in results]
    return {
        "model":           model_key,
        "total":           len(results),
        "attacks":         attacks,
        "normals":         len(results) - attacks,
        "mean_latency_ms": round(sum(lats) / len(lats), 3),
        "results":         results,
    }

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="XAI-IDS Model Server", version="1.0")


@app.get("/health")
def health():
    rss = _PROC.memory_info().rss / 1_048_576
    return {
        "status": "ok",
        "models": {
            k: {"n_features": m.n_features_in_, "classes": m.classes_.tolist()}
            for k, m in MODELS.items()
        },
        "server_rss_mb": round(rss, 2),
    }


@app.post("/kdd/upload")
async def kdd_upload(file: UploadFile):
    text = (await file.read()).decode("utf-8", errors="replace")
    try:
        X, ids = _parse_csv(text, None, _KDD_DROP, _KDD_CAT)
        return _batch("kdd", X, ids)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/unsw/upload")
async def unsw_upload(file: UploadFile):
    text = (await file.read()).decode("utf-8", errors="replace")
    try:
        X, ids = _parse_csv(text, _UNSW_COLS, _UNSW_DROP, _UNSW_CAT)
        return _batch("unsw", X, ids)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
