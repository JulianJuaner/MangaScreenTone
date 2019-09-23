import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import math
from torch.nn import init

class myVGG(nn.Module):
    def __init__(self, requires_grad=False):
        super(myVGG, self).__init__()
        original_model = models.vgg19_bn(pretrained=False)
        original_model.load_state_dict(torch.load('models/vgg19_bn-c79401a0.pth'))
        #for i in range(len(list(original_model.features.children()))):
            #print(list(original_model.features.children())[i])
        self.features = nn.Sequential(
                    # stop at conv4
                    *list(original_model.features.children())[:-4]
                )
        if requires_grad == False:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, x):
        return self.features(x)

class Illust2vec(nn.Module):
    def __init__(self, requires_grad=False):
        super(Illust2vec, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv6_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.conv6_3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.conv6_4 = nn.Conv2d(1024, 1539, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        pass
        '''
        h = self.relu(self.conv1_1(x))
        h = F.max_pool2d(h, 2, stride=2)
        h = self.relu(self.conv2_1(h))
        h = F.max_pool2d(h, 2, stride=2)
        h = self.relu(self.conv3_2(self.relu(self.conv3_1(h))))
        h = F.max_pool2d(h, 2, stride=2)
        h = self.relu(self.conv4_2(self.relu(self.conv4_1(h))))
        h = F.max_pool2d(h, 2, stride=2)
        h = self.relu(self.conv5_2(self.relu(self.conv5_1(h))))
        h = F.max_pool2d(h, 2, stride=2)
        #h = self.relu(self.conv6_4(self.relu(self.conv6_3(self.relu(self.conv6_2(self.relu(self.conv6_1(h))))))))
        #h = F.avg_pool2d(h, 7, stride=1)
        return h#.sigmoid()
        '''
class MultiLayer(nn.Module):
    def __init__(self, pth):
        super(MultiLayer, self).__init__()
        self.mean = torch.FloatTensor([164.76139251,  167.47864617,  181.13838569]).view(1, -1, 1, 1)
        illust2vec = Illust2vec()
        illust2vec.load_state_dict(torch.load(pth))
        relu = nn.ReLU()
        pad = nn.ReflectionPad2d(1)
        pool = nn.MaxPool2d(2, stride=2)
        self.pool = pool
        model_4 = [pad, illust2vec.conv1_1, relu, pool]
        model_4 += [pad, illust2vec.conv2_1, relu, pool]
        model_4 += [pad, illust2vec.conv3_1, relu] 
        model_4 += [pad, illust2vec.conv3_2, relu, pool, pad, illust2vec.conv4_1, relu]
        model_4 += [pad, illust2vec.conv4_2, relu, pool]
        model_5 = [pad, illust2vec.conv5_1, relu]
        model_6 = [pad, illust2vec.conv5_2, relu, pool, pad, illust2vec.conv6_1, relu]
        self.model_4 = nn.Sequential(*model_4)
        self.model_5 = nn.Sequential(*model_5)
        self.model_6 = nn.Sequential(*model_6)
        self.get_norm()
        
    def get_norm(self):
        mean_4 = torch.load('image/featmean_4.pt')
        mean_5 = torch.load('image/featmean_5.pt')
        mean_6 = torch.load('image/featmean_6.pt')
        std_4 = torch.load('image/featstd_4.pt')
        std_5 = torch.load('image/featstd_5.pt')
        std_6 = torch.load('image/featstd_6.pt')
        self.normalize_4 = transforms.Normalize(mean=mean_4,std=std_4)
        self.normalize_5 = transforms.Normalize(mean=mean_5,std=std_5)
        self.normalize_6 = transforms.Normalize(mean=mean_6,std=std_6)


    def forward(self, x, mode='train', is_norm=True):
        x.cuda()
        res_4 = self.model_4(x-self.mean.cuda())
        res_5 = self.model_5(res_4)
        res_6 = self.model_6(res_5)
        res_4 = self.pool(res_4)
        res_5 = self.pool(res_5)
        if is_norm:
            res_4 = self.normalize_4(res_4[0]).unsqueeze(0)
            res_5 = self.normalize_5(res_5[0]).unsqueeze(0)
            res_6 = self.normalize_6(res_6[0]).unsqueeze(0)

        if 'test' in mode:
            res = torch.cat((res_4, res_5, res_6), 1)
        else:
            res = torch.cat((res_4, res_5, res_6), 1)
        return res

class MultiScale(nn.Module):
    def __init__(self, pth):
        super(MultiScale, self).__init__()
        self.mean = torch.FloatTensor([164.76139251,  167.47864617,  181.13838569]).view(1, -1, 1, 1)
        illust2vec = Illust2vec()
        illust2vec.load_state_dict(torch.load(pth))
        relu = nn.ReLU()
        pad = nn.ReflectionPad2d(1)
        pool = nn.MaxPool2d(2, stride=2)
        self.pool = pool

        model = [pad, illust2vec.conv1_1, relu, pool]
        model += [pad, illust2vec.conv2_1, relu, pool]
        model += [pad, illust2vec.conv3_1, relu, pad, illust2vec.conv3_2, relu, pool]
        model += [pad, illust2vec.conv4_1, relu, pad, illust2vec.conv4_2, relu, pool]
        model += [pad, illust2vec.conv5_1, relu, pool]#pad, illust2vec.conv4_2, relu, pool]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x.cuda()
        res = self.model(x-self.mean.cuda())
        level_1 = F.upsample(self.pool(res), size=res.shape[2:4], mode='bilinear')
        level_2 = F.upsample(self.pool(level_1), size=res.shape[2:4], mode='bilinear')
        res = torch.cat((res, level_1, level_2), 1)
        return res

#I'm so tired for 
class Illust2vecNet(nn.Module):
    def __init__(self, pth):
        super(Illust2vecNet, self).__init__()
        self.mean = torch.FloatTensor([164.76139251,  167.47864617,  181.13838569]).view(1, -1, 1, 1)
        illust2vec = Illust2vec()
        illust2vec.load_state_dict(torch.load(pth))
        relu = nn.ReLU()
        pad = nn.ReflectionPad2d(1)
        pool = nn.MaxPool2d(2, stride=2)
        self.pool = pool
        model = [pad, illust2vec.conv1_1, relu, pool]
        model += [pad, illust2vec.conv2_1, relu, pool]
        model += [pad, illust2vec.conv3_1, relu, pad, illust2vec.conv3_2, relu, pool]
        model += [pad, illust2vec.conv4_1, relu, pad, illust2vec.conv4_2, relu, pool, pool]
        #model += [pad, illust2vec.conv5_1, relu, pad, illust2vec.conv5_2, relu, pool]
        #model += [pad, illust2vec.conv6_1, relu]#, illust2vec.conv6_2, relu]# pad, illust2vec.conv6_3, relu,
        #           pad, illust2vec.conv6_4, relu]#, nn.MaxPool2d(7, stride=1)]
        #model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)
        # self.mp = nn.MaxPool2d(4, stride=4)

    def forward(self, x, mode='train'):
        x.cuda()
        return self.model(x-self.mean.cuda())
        # return self.mp(self.model(x-self.mean))

