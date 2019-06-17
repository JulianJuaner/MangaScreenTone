from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import init
from torch.optim import lr_scheduler
from pre_networks import UnetGenerator, ResnetGenerator

relu = True

def window_stdev(X, window_size, kernel):
    X = F.pad(X, [window_size//2,window_size//2,window_size//2,window_size//2], mode='reflect')
    c1 = F.conv2d(X, kernel)
    c2 = F.conv2d(torch.pow(X,2), kernel)
    t = c2 - c1*c1
    return torch.sqrt(torch.clamp_min_(t,0))

def GaborFilter(img, gkernel):
    winsize = gkernel.shape[2]
    #input = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()
    input = img#.cuda()
    weight = torch.from_numpy(gkernel).unsqueeze(1).float().cuda()
    kernel = (torch.ones((1,1,winsize,winsize)).float() / (winsize * winsize)).cuda()
    gaborfeats = F.conv2d(F.pad(input, [winsize//2,winsize//2,winsize//2,winsize//2], mode='reflect'),weight)
    gaborfeats = gaborfeats.reshape(48,2,gaborfeats.shape[2],gaborfeats.shape[3])
    tmp = torch.sqrt(torch.sum(torch.pow(gaborfeats, 2), 1, keepdim=True))
    mean = F.conv2d(F.pad(tmp, [winsize//2,winsize//2,winsize//2,winsize//2], mode='reflect'), kernel)
    stddev = window_stdev(tmp, winsize, kernel)
    return torch.cat((mean, stddev), 0)#.cpu()#.numpy()

class GaborWavelet(nn.Module):
    def __init__(self, gkern=None, winsize=13):
        super(GaborWavelet, self).__init__()
        if gkern is None:
            gkern = torch.from_numpy(np.load("kernel2.npy")).float().unsqueeze(1)
            winsize = gkern.shape[2]
        self.pad = nn.ReflectionPad2d(winsize//2)
        # self.gabor = nn.Conv2d(1,96,kernel_size=winsize, padding=winsize//2, padding_mode='reflect', bias=False)
        self.gabor = nn.Conv2d(1,96,kernel_size=winsize, bias=False)
        self.gabor.weight.data = gkern
        self.wind = nn.Conv2d(1, 1,kernel_size=winsize, bias=False)
        nn.init.constant_(self.wind.weight.data, 1.0/(winsize*winsize))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        feats = self.gabor(self.pad(x))
        feats = feats.reshape(-1, 2, feats.shape[2], feats.shape[3])
        tmp = torch.sqrt(torch.sum(torch.pow(feats, 2), 1, keepdim=True))
        mean = self.wind(self.pad(tmp)).reshape(-1,48,feats.shape[2], feats.shape[3])
        #stddev = torch.sqrt(torch.clamp_min_(self.wind(self.pad(torch.pow(tmp,2)))-torch.pow(self.wind(self.pad(tmp)),2),0)).reshape(-1,48,feats.shape[2], feats.shape[3])
        stddev = torch.sqrt(torch.clamp(self.wind(self.pad(torch.pow(tmp,2)))-torch.pow(self.wind(self.pad(tmp)),2),min=0)).reshape(-1,48,feats.shape[2], feats.shape[3])
        return torch.cat((mean, stddev), 1)


class GaborAE(nn.Module):
    def __init__(self, in_c=1, out_c=3, dep=7, ngf=32, model='resnet'):
        super(GaborAE, self).__init__()
        if model=='unet':
            self.enc = UnetGenerator(in_c, out_c, 7, ngf)
            self.dec = UnetGenerator(out_c, in_c, 7, ngf, tanh=False)
        elif model=='resnet':
            self.enc = ResnetGenerator(in_c, out_c, ngf, n_blocks=6)
            self.dec = ResnetGenerator(out_c, in_c, ngf, n_blocks=6)

    def forward(self, x):
        inter = self.enc(x)
        # return inter[:,:3,:,:]
        recons = self.dec(inter)
        return inter, recons

class GaborVAE(nn.Module):
    def __init__(self):
        super(GaborVAE, self).__init__()
        # self.enc = nn.Sequential(nn.Conv2d(96,6,kernel_size=1,stride=1,padding=0))
        # self.dec = nn.Sequential(nn.Conv2d(3,96,kernel_size=1,stride=1,padding=0))
        # self.enc = nn.Sequential(nn.Conv2d(96,2,kernel_size=1,stride=1,padding=0), nn.Tanh())
        # self.dec = nn.Sequential(nn.Conv2d(1,96,kernel_size=1,stride=1,padding=0))
        # self.enc = nn.Sequential(nn.Conv2d(96,64,kernel_size=1,stride=1,padding=0),nn.Conv2d(64,6,kernel_size=1,stride=1,padding=0))
        # self.dec = nn.Sequential(nn.Conv2d(3,64,kernel_size=1,stride=1,padding=0),nn.Conv2d(64,96,kernel_size=1,stride=1,padding=0))
        self.enc = nn.Sequential(nn.Conv2d(96,64,kernel_size=1,stride=1,padding=0),nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0),
            nn.Conv2d(64,6,kernel_size=1,stride=1,padding=0),nn.Tanh())
        self.dec = nn.Sequential(nn.Conv2d(3,64,kernel_size=1,stride=1,padding=0),nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0),
            nn.Conv2d(64,96,kernel_size=1,stride=1,padding=0))

        if relu:
            self.enc = nn.Sequential(
                nn.Conv2d(96, 64, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 6, 3, 1, 1),
                nn.Tanh()
            )
            self.dec = self.decoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(64, 96, 3, 1, 1),
                nn.Sigmoid()
            )
        # self.enc = nn.Sequential(nn.Conv2d(96,64,kernel_size=1,stride=1,padding=0),nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0),
        #     nn.Conv2d(64,6,kernel_size=1,stride=1,padding=0))
        # self.dec = nn.Sequential(nn.Conv2d(3,64,kernel_size=1,stride=1,padding=0),nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0),
        #     nn.Conv2d(64,96,kernel_size=1,stride=1,padding=0))
        # self.enc = nn.Sequential(nn.Conv2d(96,64,kernel_size=1,stride=1,padding=0),nn.ReLU(),
        #     nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0),nn.ReLU(),
        #     nn.Conv2d(64,6,kernel_size=1,stride=1,padding=0))
        # self.dec = nn.Sequential(nn.Conv2d(3,64,kernel_size=1,stride=1,padding=0),nn.ReLU(),
        #     nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0),nn.ReLU(),
        #     nn.Conv2d(64,96,kernel_size=1,stride=1,padding=0))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std #eps.mul(std).add_(mu)
            # return torch.clamp(w + eps*std, -1,1)
        else:
            return mu

    def forward(self, x):
        inter = self.enc(x)
        w, logvar = torch.chunk(inter, 2, dim=1)
        repre = self.reparameterize(w, logvar)
        recons = self.dec(repre)
        # recons = self.dec(inter)
        reconsinter = self.enc(recons)
        return inter, recons, reconsinter, logvar, w

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.nepoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.niter_decay, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



if __name__ == '__main__':
    import time
    import cv2
    test = torch.rand((2,1,13,13)).cuda()
    # test = torch.from_numpy(cv2.imread("test1.jpg",cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(1).float().cuda()/255.0
    print(test.shape)
    # gkern = np.load("kernel.npy")
    # t1 = time.time()
    # with torch.no_grad():
    #     gabor = GaborFilter(test, gkern)#.permute(1,0,2,3)
    # t2 = time.time()
    # print("%f second."% (t2-t1))

    winsize=13
    gkern = torch.from_numpy(np.load("kernel2.npy")).float().unsqueeze(1)
    gabornet = GaborWavelet(gkern)
    gabornet.cuda()
    t1 = time.time()
    # onebatch = test[0:1]
    with torch.no_grad():
        feat = gabornet(test)
        # feat1 = gabornet(onebatch)
    t2 = time.time()
    print("%f second."% (t2-t1))
    print(feat.shape)
    # print(torch.sum(feat[0]-feat1[0]))

    # # print(torch.sum(gabor-feat))
    # feat = feat.cpu().numpy()
    # # for i, im in enumerate(feat.cpu().numpy()):
    # #     cv2.imwrite('out/%d.png' % i, im[0] * 255.0)
    #
    # from sklearn.decomposition import PCA
    # result = np.concatenate([im.reshape(1,-1) for im in feat], axis=0)
    # pca = PCA(n_components=3)
    # pca.fit(result)
    # resultPic = pca.components_.copy()
    # resultPic = resultPic.transpose().reshape((test.shape[2], test.shape[3], 3))
    # for i in range(3):
    #     tmppic = resultPic[:,:,i]
    #     resultPic[:,:,i] = (tmppic - tmppic.min()) / (tmppic.max() - tmppic.min())
    # #     cv2.normalize(tmppic,resultPic[:,:,i],0,255,dtype=cv2.NORM_MINMAX)
    # # resultPic = (resultPic - resultPic.min()) / (resultPic.max() - resultPic.min())
    # # resultPic = np.resize(resultPic, (img.shape[0], img.shape[1], 3))
    # cv2.imwrite('out/pca.png', np.uint8(resultPic*255.0))
    # # vae = GaborVAE()
    # # vae.cuda()
    # # out = vae(gabor)
    # # print(out.shape)