from dataset import *
import torch.nn as nn


BLOCKS = 8

class InpaintModel(torch.nn.Module):
    def __init__(self):
        super(InpaintModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=5, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2, 0),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3, 2, 0),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        #1/4 of original size, 256 dims
        blocks = []
        dim = 256
        dilation = 2
        use_spectral_norm = False

        for _ in range(BLOCKS):
            block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm),
                nn.InstanceNorm2d(dim, track_running_stats=False),
                nn.ReLU(True),

                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm),
                nn.InstanceNorm2d(dim, track_running_stats=False),
            )

            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1, 0)
        )

    def forward(self, x):
        x = self.encoder(x)
        #print('encoder', x.size())
        x = self.blocks(x)
        #print('blocks', x.size())
        x = self.decoder(x)
        #print('decoder', x.size())
        x = (torch.tanh(x) + 1) / 2
        return x

class doubleConv(torch.nn.Module):
    def __init__(self, input_, output):
        super(doubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_, output, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(output, output, 3, 1, 1),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.conv(x)

class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        #downscalepart.
        self.conv1 = doubleConv(5, 64)
        self.downscale1 = nn.Sequential(
            nn.MaxPool2d(2),
            doubleConv(64, 128)
        )
        self.downscale2 = nn.Sequential(
            nn.MaxPool2d(2),
            doubleConv(128, 256)
        )
        self.downscale3 = nn.Sequential(
            nn.MaxPool2d(2),
            doubleConv(256, 512)
        )
        self.downscale4 = nn.Sequential(
            nn.MaxPool2d(2),
        )
        #upscale part.
        self.conv2 = doubleConv(1024, 512)
        self.upscale2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv3 = doubleConv(512, 256)
        self.upscale3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv4 = doubleConv(256, 128)
        self.upscale4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv5 = doubleConv(128, 64)
        self.conv6 = nn.Conv2d(64, 3, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.downscale1(x1)
        x3 = self.downscale2(x2)
        x4 = self.downscale3(x3)
        x5 = self.downscale4(x4)

        x5 = self.conv2(torch.cat([self.upscale1(x5), x4], dim=1))
        x4 = 0
        x5 = self.conv3(torch.cat([self.upscale2(x5), x3], dim=1))
        x3 = 0
        x5 = self.conv4(torch.cat([self.upscale3(x5), x2], dim=1))
        x2 = 0
        x5 = self.conv5(torch.cat([self.upscale4(x5), x1], dim=1))
        x1 = 0

        x5 = self.conv6(x5)
        return x5

        
        

