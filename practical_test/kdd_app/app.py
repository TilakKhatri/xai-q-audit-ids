import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# local dev: practical_test/models/kdd | docker: overridden via MODEL_DIR env + volume mount
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(Path(__file__).resolve().parent.parent / "models" / "kdd")))

rf_model = joblib.load(MODEL_DIR / "rf_kdd_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
label_encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
LABELS_ENCODER = label_encoders['labels']
process = psutil.Process(os.getpid())


def _is_header_row(first_value) -> bool:
    """duration (col 0) is always numeric on real data rows; a header row is not."""
    try:
        float(str(first_value).strip().strip("'\""))
        return False
    except ValueError:
        return True


def load_uploaded_logs(file_storage):
    """Parse a .csv/.txt KDD-style log: works with or without a header row,
    and with or without a trailing ground-truth label column."""
    raw = pd.read_csv(file_storage, header=None, dtype=str, skip_blank_lines=True, engine='python')
    raw = raw.apply(lambda col: col.str.strip().str.strip("'\""))

    if _is_header_row(raw.iloc[0, 0]):
        raw = raw.iloc[1:].reset_index(drop=True)

    n_features = len(feature_columns)
    if raw.shape[1] == n_features + 1:
        raw.columns = feature_columns + ['actual_label']
        actual_labels = raw['actual_label'].str.rstrip('.')
        df = raw[feature_columns].copy()
    elif raw.shape[1] == n_features:
        raw.columns = feature_columns
        actual_labels = None
        df = raw.copy()
    else:
        raise ValueError(
            f"Expected {n_features} columns (or {n_features + 1} with a label column), got {raw.shape[1]}"
        )

    for col in df.columns:
        if col not in CATEGORICAL_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.drop(columns=CATEGORICAL_COLS).isna().any().any():
        raise ValueError("Non-numeric values found in numeric feature columns")

    return df, actual_labels


@app.route('/')
def home():
    return render_template('upload.html')  # Simple upload page


@app.route('/predict', methods=['POST'])
def predict():
    t_start = time.perf_counter()
    cpu_before = process.cpu_times()
    mem_before = process.memory_info().rss

    try:
        df, actual_labels = load_uploaded_logs(request.files['file'])
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    t_read = time.perf_counter()

    # map any unseen category to a known one before encoding (avoids ValueError)
    for col in CATEGORICAL_COLS:
        known = set(label_encoders[col].classes_)
        df[col] = df[col].where(df[col].isin(known), label_encoders[col].classes_[0])
        df[col] = label_encoders[col].transform(df[col])

    X = scaler.transform(df[feature_columns])
    t_preprocess = time.perf_counter()

    predictions = rf_model.predict(X)
    probabilities = rf_model.predict_proba(X)
    t_predict = time.perf_counter()

    pred_labels = LABELS_ENCODER.inverse_transform(predictions)
    pred_labels = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in pred_labels]
    pred_labels = [lbl.rstrip('.') for lbl in pred_labels]

    results = []
    for i, (label, proba) in enumerate(zip(pred_labels, probabilities)):
        is_attack = not label.startswith('normal')
        entry = {
            'Row': i + 1,
            'Actual_Class': 'Attack' if is_attack else 'Normal',
            'Predicted_Label': label,
            'Confidence': f"{np.max(proba):.2%}",
            'Remark': 'Attack Detected - Investigate!' if is_attack else 'Normal Traffic'
        }
        if actual_labels is not None:
            actual = actual_labels.iloc[i]
            entry['Ground_Truth_Label'] = actual
            entry['Prediction_Correct'] = (label == actual)
        results.append(entry)
    t_end = time.perf_counter()

    cpu_after = process.cpu_times()
    mem_after = process.memory_info().rss

    response = {
        'total_logs': len(results),
        'attacks_detected': sum(1 for r in results if r['Actual_Class'] == 'Attack'),
        'has_ground_truth': actual_labels is not None,
        'results': results,
        'timing_ms': {
            'file_read_ms': round((t_read - t_start) * 1000, 2),
            'preprocess_ms': round((t_preprocess - t_read) * 1000, 2),
            'inference_ms': round((t_predict - t_preprocess) * 1000, 2),
            'postprocess_ms': round((t_end - t_predict) * 1000, 2),
            'total_ms': round((t_end - t_start) * 1000, 2),
        },
        'resource_usage': {
            'cpu_time_user_s': round(cpu_after.user - cpu_before.user, 4),
            'cpu_time_system_s': round(cpu_after.system - cpu_before.system, 4),
            'memory_rss_mb': round(mem_after / (1024 ** 2), 2),
            'memory_delta_mb': round((mem_after - mem_before) / (1024 ** 2), 2),
        },
    }
    if actual_labels is not None:
        response['accuracy'] = round(sum(r['Prediction_Correct'] for r in results) / len(results), 4)

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)