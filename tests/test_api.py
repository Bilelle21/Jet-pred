import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from jetpred import config
from jetpred.serve import MAX_BATCH_SIZE, LoadedModel, create_app

FEATURES = [f"feature_{i}" for i in range(config.N_FEATURES)]


class FakeModel:
    def predict(self, frame):
        probs = np.full((len(frame), len(config.CLASSES)), 0.1)
        probs[:, 2] = 0.6
        output = pd.DataFrame(probs, columns=[f"prob_{c}" for c in config.CLASSES])
        output.insert(0, "prediction", config.CLASSES[2])
        return output


@pytest.fixture
def client():
    app = create_app(loader=lambda: LoadedModel(FakeModel(), FEATURES, "test"))
    with TestClient(app) as test_client:
        yield test_client


def make_jet():
    return {name: 0.5 for name in FEATURES}


def test_health(client):
    assert client.get("/health").status_code == 200


def test_ready_reports_version(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["model_version"] == "test"


def test_predict_returns_one_prediction_per_jet(client):
    response = client.post("/predict", json={"jets": [make_jet(), make_jet()]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    first = body["predictions"][0]
    assert first["prediction"] in config.CLASSES
    assert abs(sum(first["probabilities"].values()) - 1) < 1e-6


def test_missing_feature_is_rejected(client):
    jet = make_jet()
    del jet[FEATURES[0]]
    response = client.post("/predict", json={"jets": [jet]})
    assert response.status_code == 422
    assert FEATURES[0] in response.json()["detail"]["missing"]


def test_empty_batch_is_rejected(client):
    assert client.post("/predict", json={"jets": []}).status_code == 422


def test_oversized_batch_is_rejected(client):
    jets = [make_jet()] * (MAX_BATCH_SIZE + 1)
    assert client.post("/predict", json={"jets": jets}).status_code == 422