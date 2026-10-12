import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from jetpred.model import JetClassifier


class JetClassifierPyfunc(mlflow.pyfunc.PythonModel):
    """Scaler + network + label decoding, packaged as a single MLflow model."""

    def __init__(self, feature_names, classes, hidden_sizes=(64, 32, 32)):
        self.feature_names = list(feature_names)
        self.classes = list(classes)
        self.hidden_sizes = tuple(hidden_sizes)

    def load_context(self, context):
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.model = JetClassifier(hidden_sizes=self.hidden_sizes, n_classes=len(self.classes))
        self.model.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu"))
        self.model.eval()

    def predict(self, context, model_input, params=None):
        # Select by name so that column order in the request never matters
        X = self.scaler.transform(model_input[self.feature_names]).astype(np.float32)
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.from_numpy(X)), dim=1).numpy()

        output = pd.DataFrame(probs, columns=[f"prob_{c}" for c in self.classes])
        output.insert(0, "prediction", [self.classes[i] for i in probs.argmax(axis=1)])
        return output