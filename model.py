import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import torch.nn.functional as F


class DoubConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)

        )
        self.conv2 = nn.Sequential(

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x1 = x*torch.tanh(F.softplus(x))
        x = self.conv2(x1)
        x = x*torch.tanh(F.softplus(x))
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, kernal_channels=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.conv = nn.ModuleList()
        self.deconv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for channel in kernal_channels:
            self.conv.append(DoubConv(in_channels, channel))
            in_channels = channel

        for channel in reversed(kernal_channels):
            self.deconv.append(nn.ConvTranspose2d(
                channel*2, channel, 2, 2, bias=False))
            self.deconv.append(DoubConv(channel*2, channel))
        self.bottleneck = DoubConv(kernal_channels[-1], kernal_channels[-1]*2)
        self.output = nn.Conv2d(
            kernal_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for convs in self.conv:
            x = convs(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[:: -1]
        for i in range(0, len(self.deconv), 2):
            x = self.deconv[i](x)
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                # ignoring batch size and the no of channels
                x = tf.resize(x, size=skip_connection.shape[2:])
            concat = torch.cat((skip_connection, x), dim=1)
            x = self.deconv[i+1](concat)
        return self.output(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = Unet(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
