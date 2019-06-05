from dataset import *
import torch.nn as nn


BLOCKS = 8

class InpaintModel(torch.nn.Module):
    def __init__(self):
        super(InpaintModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
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
                nn.utils.spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
                nn.InstanceNorm2d(dim, track_running_stats=False),
                nn.ReLU(True),

                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
                nn.InstanceNorm2d(dim, track_running_stats=False),
            )

            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, 1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2, 1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1, 0)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.decoder(x)
        return x
