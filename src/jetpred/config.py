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