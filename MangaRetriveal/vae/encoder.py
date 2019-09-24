import torch
from torch import nn
import torch.nn.functional as F

CHANNEL = 64
inchannel = 2048
isVAE = True

# The feature map. Same as the convolution process in the RPN head. 
# Return a response feature map with the input and the query kernel.
class FeatureMap(torch.nn.Module):
    def __init__(self, kern, in_channels=2048):
        super(FeatureMap, self).__init__()
        kern.requires_grad = False
        rh = kern.shape[2] // 2 * 2 + 1
        rw = kern.shape[3] // 2 * 2 + 1
        kern = F.pad(kern, (0, rw - kern.shape[3], 0, rh - kern.shape[2]), 'reflect')
        self.pad = nn.ReflectionPad2d((kern.shape[3] // 2, kern.shape[3] // 2, kern.shape[2] // 2, kern.shape[2] // 2))
        self.sim = nn.Conv2d(in_channels, 1, kernel_size=(kern.shape[3], kern.shape[2]), stride=1, bias=False)
        self.sim.weight = nn.Parameter(kern)
        self.tnorm = nn.Conv2d(in_channels, 1, kernel_size=(kern.shape[3], kern.shape[2]), stride=1, bias=False)
        self.tnorm.weight = nn.Parameter(torch.ones_like(kern, requires_grad=False))
        self.knorm = kern.pow(2).sum().pow(0.5)

    def forward(self, x):
        t = self.sim(self.pad(x)).div(self.tnorm(self.pad(x.pow(2))).pow(0.5)*self.knorm+1e-8)
        return t

class FeatureEncoder(torch.nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(inchannel,512,kernel_size=1,stride=1,padding=0),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,CHANNEL,kernel_size=1,stride=1,padding=0),
            #nn.InstanceNorm2d(CHANNEL),
            #nn.ReLU(),
            #nn.Tanh()
            )

    def forward(self, x):
        inter = self.enc(x)
        #w= torch.chunk(inter, 2, dim=1)
        #repre = self.reparameterize(w, logvar)
        return inter

class FeatureEncoderNorm(torch.nn.Module):
    def __init__(self):
        super(FeatureEncoderNorm, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(inchannel,256,kernel_size=1,stride=1,padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,CHANNEL,kernel_size=1,stride=1,padding=0),
            #nn.InstanceNorm2d(CHANNEL),
            nn.ReLU(),
            #nn.Sigmoid()
            )

    def forward(self, x):
        inter = self.enc(x)
        #w= torch.chunk(inter, 2, dim=1)
        #repre = self.reparameterize(w, logvar)
        return inter

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,128,kernel_size=3,stride=2,padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,CHANNEL*2,kernel_size=3,stride=1,padding=1),
            #nn.InstanceNorm2d(CHANNEL),
            nn.Tanh()
            )

        self.dec = nn.Sequential(
            nn.Conv2d(CHANNEL,128,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
             nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid()
            )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std #eps.mul(std).add_(mu)
            # return torch.clamp(w + eps*std, -1,1)
        else:
            return mu

    def changeImage(self):
        pass

    def forward(self, x):
        inter = self.enc(x)
        w, logvar = torch.chunk(inter, 2, dim=1)
        repre = self.reparameterize(w, logvar)
        recons = self.dec(repre)
        # recons = self.dec(inter)
        reconsinter = self.enc(recons)
        return inter, recons, reconsinter, logvar, w, repre