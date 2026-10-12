import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from jetpred import config

EXPORT_DIR = Path("build/model")

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
version = MlflowClient().get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS).version

if EXPORT_DIR.exists():
    shutil.rmtree(EXPORT_DIR)
EXPORT_DIR.parent.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    downloaded = Path(mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{config.MODEL_NAME}/{version}", dst_path=tmp,
    ))
    model_root = next(downloaded.rglob("MLmodel")).parent
    shutil.copytree(model_root, EXPORT_DIR)

(EXPORT_DIR / "VERSION").write_text(str(version))
print(f"Exported {config.MODEL_NAME} v{version} to {EXPORT_DIR}")