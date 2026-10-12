import time
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from jetpred import config
from jetpred.data import load_data, split_data
from jetpred.onnx_export import build_inference_model, export_onnx

OUT_DIR = Path("build/onnx")
BATCH_SIZES = [1, 100, 1000]


def time_calls(fn, n_iter, n_warmup=20):
    for _ in range(n_warmup):
        fn()
    timings = []
    for _ in range(n_iter):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)
    return np.median(timings), np.percentile(timings, 95)


def make_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def main():
    torch.set_num_threads(1)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    served = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}")
    inner = served.unwrap_python_model()

    X, y = load_data()
    _, _, X_test, _, _, y_test = split_data(X, y)
    X_np = X_test[inner.feature_names].to_numpy(dtype=np.float32)

    torch_model = build_inference_model(inner.model, inner.scaler)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = export_onnx(torch_model, OUT_DIR / "model_fp32.onnx", config.N_FEATURES)
    int8_path = OUT_DIR / "model_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    sessions = {"onnx_fp32": make_session(fp32_path), "onnx_int8": make_session(int8_path)}

    # 1. Correctness on the full test set, before any timing
    labels = np.array(config.CLASSES)
    y_true = y_test.to_numpy()
    with torch.inference_mode():
        reference = torch_model(torch.from_numpy(X_np)).numpy()
    print("Accuracy on the full test set")
    print(f"  {'torch':<10} {(labels[reference.argmax(1)] == y_true).mean():.4f}")
    for name, session in sessions.items():
        probs = session.run(None, {"input": X_np})[0]
        accuracy = (labels[probs.argmax(1)] == y_true).mean()
        max_diff = np.abs(probs - reference).max()
        print(f"  {name:<10} {accuracy:.4f}   (max probability difference vs torch: {max_diff:.2e})")

    # 2. Latency
    rows = []
    with torch.inference_mode():
        for batch_size in BATCH_SIZES:
            frame = X_test.iloc[:batch_size]
            array = X_np[:batch_size]
            tensor = torch.from_numpy(array)
            n_iter = 200 if batch_size < 1000 else 50

            candidates = {
                "mlflow_pyfunc": lambda: served.predict(frame),
                "pyfunc_no_schema": lambda: inner.predict(None, frame),
                "torch": lambda: torch_model(tensor),
                "onnx_fp32": lambda: sessions["onnx_fp32"].run(None, {"input": array}),
                "onnx_int8": lambda: sessions["onnx_int8"].run(None, {"input": array}),
            }
            for name, fn in candidates.items():
                median, p95 = time_calls(fn, n_iter)
                rows.append({
                    "batch": batch_size,
                    "method": name,
                    "median_ms": median,
                    "p95_ms": p95,
                    "us_per_jet": median * 1000 / batch_size,
                })

    results = pd.DataFrame(rows)
    print()
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(OUT_DIR / "benchmark.csv", index=False)

    sizes = {p.name: p.stat().st_size / 1024 for p in (fp32_path, int8_path)}
    print("\nModel sizes (KB):", {k: round(v, 1) for k, v in sizes.items()})


if __name__ == "__main__":
    main()