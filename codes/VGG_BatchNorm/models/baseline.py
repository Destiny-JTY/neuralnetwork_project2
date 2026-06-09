"""A compact CNN baseline for CIFAR-10."""

from torch import nn

try:
    from ..utils.nn import init_weights_
except ImportError:
    from utils.nn import init_weights_


class BaselineCNN(nn.Module):
    """Simple Conv-ReLU-Pool network followed by fully connected layers."""

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Linear(256, num_classes),
        )

        if init_weights:
            for module in self.modules():
                init_weights_(module)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.reshape(x.size(0), -1))
