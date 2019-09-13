
# how to use
# criterionP = PercLoss(stage=opt.stage).to(self.device)
# loss = criterionP(output, target)
import torch
from torch import nn
import torch.nn.functional as F

class PercLoss(nn.Module):
    def __init__(self, stage=3, use_L2=True):#, gpu_ids=[]):
        super(PercLoss, self).__init__()

        cuda = torch.cuda.is_available()
        i2vnet = Illust2vec()
        
        # init_net(i2vnet, gpu_ids=gpu_ids)
        i2vnet.load_state_dict(torch.load('./models/illust2vec_tag_ver200_2.pth'))
        self.model = i2vnet.get_stages(stage)
        if cuda:
            self.model = self.model.cuda()
            self.model=nn.DataParallel(self.model)
        self.use_L2 = use_L2

    def __call__(self, input, target, batch_mean=True):
        if self.use_L2:
            diff = (self.model(input.repeat(1,3,1,1)) - self.model(target.repeat(1,3,1,1))) ** 2
        else:
            diff = torch.abs(self.model(input.repeat(1,3,1,1)) - self.model(target.repeat(1,3,1,1)))

        if batch_mean:
            return torch.mean(diff)
        else:
            return torch.mean(torch.mean(torch.mean(diff, dim=1), dim=2), dim=3)

class Illust2vec(nn.Module):
    def __init__(self, requires_grad=False):
        super(Illust2vec, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6_3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6_4 = nn.Conv2d(1024, 1539, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def get_stages(self, stage=5):
        levels = [[self.conv1_1, nn.ReLU(), nn.MaxPool2d(2, stride=2)],
                  [self.conv2_1, nn.ReLU(), nn.MaxPool2d(2, stride=2)],
                  [self.conv3_1, nn.ReLU(), self.conv3_2, nn.ReLU(), nn.MaxPool2d(2, stride=2)],
                  [self.conv4_1, nn.ReLU(), self.conv4_2, nn.ReLU(), nn.MaxPool2d(2, stride=2)],
                  [self.conv5_1, nn.ReLU(), self.conv5_2, nn.ReLU(), nn.MaxPool2d(2, stride=2)],
                  [self.conv6_1, nn.ReLU(), self.conv6_2, nn.ReLU(), self.conv6_3, nn.ReLU(), self.conv6_4, nn.ReLU()],
                  [nn.MaxPool2d(7, stride=1), nn.Sigmoid()]]#nn.AdaptiveAvgPool(1)
        models = []
        for i in range(stage):
            models += levels[i] 
        model = nn.Sequential(*models)
        return model

    def forward(self, x):
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
        h = self.relu(self.conv6_4(self.relu(self.conv6_3(self.relu(self.conv6_2(self.relu(self.conv6_1(h))))))))
        h = F.avg_pool2d(h, 7, stride=1)
        return h.sigmoid()