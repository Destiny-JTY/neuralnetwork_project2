"""A compact CNN baseline for CIFAR-10."""

from torch import nn

try:
    from ..utils.nn import init_weights_
except ImportError:
    from utils.nn import init_weights_


class BaselineCNN(nn.Module):
    """Simple Conv-Activation-Pool network followed by fully connected layers."""

    def __init__(
        self,
        inp_ch=3,
        num_classes=10,
        channels=(32, 64, 128),
        hidden_width=256,
        activation="relu",
        use_batch_norm=False,
        dropout=0.0,
        init_weights=True,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("BaselineCNN expects exactly three channel widths.")
        if hidden_width < 1:
            raise ValueError("hidden_width must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.channels = tuple(channels)
        self.hidden_width = hidden_width
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout

        feature_layers = []
        in_channels = inp_ch
        for out_channels in self.channels:
            feature_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            if use_batch_norm:
                feature_layers.append(nn.BatchNorm2d(out_channels))
            feature_layers.append(_make_activation(activation))
            feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.features = nn.Sequential(*feature_layers)

        classifier_layers = [
            nn.Linear(self.channels[-1] * 4 * 4, hidden_width),
            _make_activation(activation),
        ]
        if dropout > 0.0:
            classifier_layers.append(nn.Dropout(p=dropout))
        classifier_layers.append(nn.Linear(hidden_width, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        if init_weights:
            for module in self.modules():
                init_weights_(module)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.reshape(x.size(0), -1))


def _make_activation(name):
    activations = {
        "relu": lambda: nn.ReLU(True),
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True),
        "gelu": nn.GELU,
    }
    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc
