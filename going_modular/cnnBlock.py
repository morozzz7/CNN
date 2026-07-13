from torch import nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, hidden_units, dropout_p=0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units,
            kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units * 2,
            kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 2, out_channels=hidden_units * 2,
            kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 2, out_channels=hidden_units * 4,
            kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 4, out_channels=hidden_units * 4,
            kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 4, out_channels=hidden_units * 8,
            kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 8, out_channels=hidden_units * 8,
            kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 8),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
