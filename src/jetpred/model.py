from torch import nn

from jetpred import config


class JetClassifier(nn.Module):
    """MLP :  16 features to predict 5 classes."""

    def __init__(
        self,
        n_features=config.N_FEATURES,
        hidden_sizes=(64, 32, 32),
        n_classes=None,
        dropout=0.1,
    ):
        super().__init__()
        if n_classes is None:
            n_classes = len(config.CLASSES)
        layers = []
        in_size = n_features
        for hidden_size in hidden_sizes:
            layers += [
                nn.Linear(in_size, hidden_size), 
                nn.ReLU(),                      
                nn.Dropout(dropout),            
            ]
            in_size = hidden_size

        layers.append(nn.Linear(in_size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)