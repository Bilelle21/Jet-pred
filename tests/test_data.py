import pytest

from jetpred import config
from jetpred.data import load_data, prepare_data, split_data


@pytest.fixture(scope="module")
def raw():
    return load_data()


@pytest.fixture(scope="module")
def prepared():
    return prepare_data()


def test_schema(raw):
    X, y = raw
    assert X.shape[1] == config.N_FEATURES
    assert len(X) == len(y)


def test_no_missing_values(raw):
    X, y = raw
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_expected_classes(raw):
    _, y = raw
    assert sorted(y.unique()) == config.CLASSES


def test_splits_do_not_overlap(raw):
    """Check data leaks"""
    X, y = raw
    X_train, X_val, X_test, *_ = split_data(X, y)
    train, val, test = set(X_train.index), set(X_val.index), set(X_test.index)
    assert not train & val
    assert not train & test
    assert not val & test
    assert len(train) + len(val) + len(test) == len(X)
