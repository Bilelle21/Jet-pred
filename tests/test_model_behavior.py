import mlflow
import numpy as np
import pytest

from jetpred import config
from jetpred.data import load_data, split_data

pytestmark = pytest.mark.model

PROB_COLUMNS = [f"prob_{c}" for c in config.CLASSES]


@pytest.fixture(scope="module")
def model():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}")


@pytest.fixture(scope="module")
def test_set():
    X, y = load_data()
    _, _, X_test, _, _, y_test = split_data(X, y)
    return X_test.iloc[:20_000], y_test.iloc[:20_000]


def test_output_format(model, test_set):
    X, _ = test_set
    out = model.predict(X.head(10))
    assert set(out["prediction"]) <= set(config.CLASSES)
    assert np.allclose(out[PROB_COLUMNS].sum(axis=1), 1, atol=1e-5)


def test_deterministic(model, test_set):
    X, _ = test_set
    first = model.predict(X.head(100))[PROB_COLUMNS].to_numpy()
    second = model.predict(X.head(100))[PROB_COLUMNS].to_numpy()
    assert np.array_equal(first, second)


def test_batch_matches_single(model, test_set):
    X, _ = test_set
    batch = model.predict(X.head(20))[PROB_COLUMNS].to_numpy()
    single = np.vstack([model.predict(X.iloc[[i]])[PROB_COLUMNS].to_numpy() for i in range(20)])
    assert np.allclose(batch, single, atol=1e-5)


def test_column_order_does_not_matter(model, test_set):
    X, _ = test_set
    sample = X.head(50)
    reordered = sample[sample.columns[::-1]]
    assert np.allclose(
        model.predict(sample)[PROB_COLUMNS].to_numpy(),
        model.predict(reordered)[PROB_COLUMNS].to_numpy(),
        atol=1e-6,
    )


def test_accuracy_above_threshold(model, test_set):
    X, y = test_set
    accuracy = (model.predict(X)["prediction"].to_numpy() == y.to_numpy()).mean()
    assert accuracy >= config.MIN_TEST_ACCURACY