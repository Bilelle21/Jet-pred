import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from jetpred import config



def load_data():
    """Download and read the data ht4ml"""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(
        config.DATASET_NAME,
        as_frame=True,             
        data_home=config.DATA_DIR, 
    )
    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(str)
    return X, y

def split_data(X, y):
    """Split the data between train, test and validation"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,              
        random_state=config.SEED, 
    )
    val_fraction = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_fraction,
        stratify=y_train_val,
        random_state=config.SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def encode_labels(y):
    """Encode the labels"""
    mapping = {name: index for index, name in enumerate(config.CLASSES)}
    return y.map(mapping).to_numpy(dtype=np.int64)

def fit_scaler(X_train):
    """Scaler for our train set"""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def save_scaler(scaler, path=None):
    """Save the scaler to use for inference"""
    path = path or config.ARTIFACTS_DIR / "scaler.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return path


def prepare_data():
    """Complete pipeline for our data"""
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    scaler = fit_scaler(X_train) 

    return {
        "X_train": scaler.transform(X_train).astype(np.float32),
        "X_val": scaler.transform(X_val).astype(np.float32),
        "X_test": scaler.transform(X_test).astype(np.float32),
        "y_train": encode_labels(y_train),
        "y_val": encode_labels(y_val),
        "y_test": encode_labels(y_test),
        "scaler": scaler,
        "feature_names": list(X.columns),
    }