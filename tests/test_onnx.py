import numpy as np
import onnxruntime as ort
import torch
from sklearn.preprocessing import StandardScaler

from jetpred import config
from jetpred.model import JetClassifier
from jetpred.onnx_export import build_inference_model, export_onnx


def test_onnx_matches_torch(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(64, config.N_FEATURES)).astype(np.float32)
    scaler = StandardScaler().fit(X)
    model = build_inference_model(JetClassifier(), scaler)

    path = export_onnx(model, tmp_path / "model.onnx", config.N_FEATURES)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    onnx_probs = session.run(None, {"input": X})[0]
    with torch.no_grad():
        torch_probs = model(torch.from_numpy(X)).numpy()

    assert onnx_probs.shape == (64, len(config.CLASSES))
    assert np.allclose(onnx_probs, torch_probs, atol=1e-5)