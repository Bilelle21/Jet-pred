import torch
from torch import nn


class InferenceModel(nn.Module):
    """Standardisation + classifier + softmax in a single exportable graph"""

    def __init__(self, classifier, mean, scale):
        super().__init__()
        self.classifier = classifier
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return torch.softmax(self.classifier((x - self.mean) / self.scale), dim=1)


def build_inference_model(classifier, scaler):
    return InferenceModel(classifier, scaler.mean_, scaler.scale_).eval()


def export_onnx(model, path, n_features):
    dummy_input = torch.zeros(1, n_features)
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path