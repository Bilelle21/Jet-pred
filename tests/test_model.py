import torch
from torch import nn

from jetpred import config
from jetpred.model import JetClassifier


def test_output_shape():
    """For 8 jets, we have 5 times 8 logits"""
    model = JetClassifier()
    out = model(torch.randn(8, config.N_FEATURES))
    assert out.shape == (8, len(config.CLASSES))


def test_can_overfit_small_batch():
    """Our model learn well."""
    torch.manual_seed(0)
    X = torch.randn(32, config.N_FEATURES)
    y = torch.randint(0, len(config.CLASSES), (32,))
    model = JetClassifier(dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(300):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1