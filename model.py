import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        orig_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + orig_x
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale1 = nn.Parameter(torch.randn(6, 6))
        self.bias1 = nn.Parameter(torch.randn(4, 6, 6))
        self.conv1 = nn.Conv2d(4, 64, 3, padding='same')
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 5, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(64, 3, 9, padding='same')
    
    def forward(self, x):
        x = x.type(torch.float32)
        x = (x - 127.5) / 127.5
        x = x.reshape(-1, 6, 6, 4)
        x = x.permute(0, 3, 1, 2)
        x = torch.mul(x, self.scale1) + self.bias1
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = x * 150
        x = x + 127.5
        return x
