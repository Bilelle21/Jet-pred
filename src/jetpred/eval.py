import argparse
from types import SimpleNamespace

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)

from jetpred import config
from jetpred.data import load_data, split_data
from jetpred.pyfunc_model import JetClassifierPyfunc


def best_training_run():
    runs = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT_NAME],
        filter_string="metrics.best_val_loss > 0",
        order_by=["metrics.best_val_loss ASC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No completed training run found.")
    return runs.iloc[0]["run_id"]


def compute_metrics(y_true, output, classes):
    y_pred = output["prediction"].to_numpy()
    probs = output[[f"prob_{c}" for c in classes]].to_numpy()
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)

    metrics = {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_f1_macro": report["macro avg"]["f1-score"],
        "test_auc_macro": roc_auc_score(y_true, probs, multi_class="ovr", labels=classes),
    }
    for i, c in enumerate(classes):
        metrics[f"test_f1_{c}"] = report[c]["f1-score"]
        metrics[f"test_auc_{c}"] = roc_auc_score(y_true == c, probs[:, i])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Package, evaluate and register a trained model")
    parser.add_argument("--run-id", help="Training run to use (default: lowest validation loss)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    run_id = args.run_id or best_training_run()

    artifacts = {
        "weights": mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model.pt"),
        "scaler": mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.joblib"),
    }

    X, y = load_data()
    _, _, X_test, _, _, y_test = split_data(X, y)

    preview = JetClassifierPyfunc(X.columns, config.CLASSES)
    preview.load_context(SimpleNamespace(artifacts=artifacts))
    example = X_test.head(5)
    signature = infer_signature(example, preview.predict(None, example))

    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=JetClassifierPyfunc(X.columns, config.CLASSES),
            artifacts=artifacts,
            signature=signature,
            input_example=example,
            registered_model_name=config.MODEL_NAME,
        )

        model = mlflow.pyfunc.load_model(model_info.model_uri)
        output = model.predict(X_test)
        metrics = compute_metrics(y_test, output, config.CLASSES)
        mlflow.log_metrics(metrics)

        display = ConfusionMatrixDisplay.from_predictions(
            y_test, output["prediction"], labels=config.CLASSES,
            normalize="true", values_format=".2f",
        )
        mlflow.log_figure(display.figure_, "confusion_matrix.png")

    version = model_info.registered_model_version
    print(f"Test accuracy: {metrics['test_accuracy']:.4f} | macro AUC: {metrics['test_auc_macro']:.4f}")

    if metrics["test_accuracy"] >= config.MIN_TEST_ACCURACY:
        MlflowClient().set_registered_model_alias(config.MODEL_NAME, config.MODEL_ALIAS, version)
        print(f"Version {version} promoted to @{config.MODEL_ALIAS}")
    else:
        print(f"Version {version} registered but not promoted (threshold {config.MIN_TEST_ACCURACY})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()