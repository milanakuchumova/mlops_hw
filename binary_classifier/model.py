from torch import nn


class BasicBlockNet(nn.Module):
    def __init__(self, n_classes, size_h, size_w):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
        )
        self.skip_conn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)
        )
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            in_features=32 * (size_h // 8) * (size_w // 8), out_features=n_classes
        )

    def forward(self, x):
        skip_conn_x = x
        out = self.net(x)
        out += self.skip_conn(skip_conn_x)
        out = self.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
