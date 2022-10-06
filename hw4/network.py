import torch.nn as nn


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class Hallusnator(nn.Module):

    def __init__(self, x_dim=1800, out_dim=1600):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(x_dim, out_dim),
            nn.ReLU(),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.fc(x)
        return x.view(x.size(0))

class Discriminator(nn.Module):
    def __init__(self, input_len=1600, hidden_len=16, output_len=1):
        super(Discriminator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_len, hidden_len),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_len, output_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x