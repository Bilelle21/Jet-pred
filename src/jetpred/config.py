from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"          
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" 



# Constants
DATASET_NAME = "hls4ml_lhc_jets_hlf"
N_FEATURES = 16
CLASSES = ["g", "q", "t", "w", "z"] 
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Expriments
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "jet-pred"


MODEL_NAME = "jet-classifier"
MODEL_ALIAS = "champion"
MIN_TEST_ACCURACY = 0.74