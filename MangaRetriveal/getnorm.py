from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.autograd import Variable
from datasets import DataDataset
import math
import tensorboardX
from models.networks import Illust2vecNet, myVGG, MultiScale, MultiLayer

# import tqdm
# from progressbar import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--zdim', type=int, default=3, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--start', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='gabor_vae2',  help='output folder')
parser.add_argument('--model', type=str, default='unet',  help='model type')
parser.add_argument('--initmodel', type=str, default='',  help='initialized model path')
parser.add_argument('--lr', type=float, default=0.0001,  help='learning rate')
parser.add_argument('--dataset', type=str, default='../../../data/manga',  help='dataset mode')
parser.add_argument('--cropSize', type=int, default=256, help='input batch size')
parser.add_argument('--lr_policy', type=str, default='step', help='input batch size')
parser.add_argument('--niter', type=int, default=200, help='input batch size')
parser.add_argument('--niter_decay', type=int, default=200, help='input batch size')
parser.add_argument('--preprocess', type=str, default='crop', help='input batch size')
parser.add_argument('--no_flip', action='store_true', help='input batch size')
parser.add_argument('--blocks', type=int, default=3, help='number of epochs to train for')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
base = 2**4
chnl = 512
dirName = 'simline'
dataset = DataDataset(args.dataset, base=base, chnls=chnl, mode='img', name=dirName)
# dataset = DataDataset('data/p11')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=args.workers)
size=args.cropSize
ll = len(dataset)

def test():
    # pbar = ProgressBar().start()
    # sums = 0 # 0.0358
    # # mins = torch.zeros(ll) # 0
    # maxs = torch.zeros(ll) # 0.4022
    # pixs=0
    # for i, data in enumerate(dataloader, 0):
    #     # print(data[1])
    #     X = Variable(data[0].cuda(), requires_grad=False)
    #     # Forward
    #     gaborfeat = gaborext((X+1)/2)
    #     pixs += gaborfeat.shape[2]*gaborfeat.shape[3]*chnl
    #     sums += gaborfeat.sum()
    #     # mins[i] = gaborfeat.min()
    #     maxs[i] = gaborfeat.max()
    #     gaborfeat = None
    #     torch.cuda.empty_cache()
    # print(sums.sum()/pixs, maxs.max())

    # stds = 0 # 0.0705
    # pixs = 0
    # mean = 0.0358
    # for i, data in enumerate(dataloader, 0):
    #     X = Variable(data[0].cuda(), requires_grad=False)
    #     # Forward
    #     gaborfeat = gaborext((X+1)/2)
    #     pixs += gaborfeat.shape[2]*gaborfeat.shape[3]*chnl
    #     stds += ((gaborfeat-mean)**2).sum()
    #     gaborfeat = None
    #     torch.cuda.empty_cache()
    # print((stds.sum()/pixs).sqrt())

    # max = 0.4022
    # min = 0
    # mean = 0.0358
    # std = 0.0705

    sums = torch.zeros(chnl).cuda() # 0.0358
    mins = torch.ones(chnl).cuda() # 0
    maxs = torch.zeros(chnl).cuda() # 0.4022
    pixs = 0
    # pbar = tqdm(total=ll)
    for i, data in enumerate(dataloader, 0):
        # print(data[1])
        X = Variable(data[0].cuda(), requires_grad=False)
        # print(X.max())
        # Forward
        # gaborfeat = gaborext((X+1)/2).reshape(chnl,-1)
        gaborfeat = gaborext(X).reshape(chnl,-1)
        pixs += gaborfeat.shape[1]
        sums += gaborfeat.sum(dim=1)
        # mins = torch.min(gaborfeat.min(dim=1)[0], mins)
        # maxs = torch.max(gaborfeat.max(dim=1)[0], maxs)
        gaborfeat = None
        torch.cuda.empty_cache()
        # pbar.update(1)
    # print(sums/pixs, maxs, mins)
    print(sums/pixs)
    torch.save(sums/pixs, "featmean_6.pt")
    # max
    # [0.3839, 0.4022, 0.3839, 0.4014, 0.2408, 0.2535, 0.2389, 0.2552, 0.3428,
    #     0.2841, 0.3492, 0.2812, 0.3279, 0.2871, 0.3253, 0.2907, 0.3158, 0.2549,
    #     0.3034, 0.2524, 0.2083, 0.2006, 0.1972, 0.1839, 0.2182, 0.1887, 0.2183,
    #     0.1886, 0.2761, 0.2402, 0.2762, 0.2389, 0.2846, 0.2828, 0.2863, 0.2779,
    #     0.2845, 0.2885, 0.2835, 0.2916, 0.3059, 0.2549, 0.2946, 0.2528, 0.1chnl9,
    #     0.2006, 0.1847, 0.1840, 0.1575, 0.1654, 0.1577, 0.1656, 0.1300, 0.1478,
    #     0.1315, 0.1469, 0.1988, 0.1384, 0.1989, 0.1391, 0.1677, 0.1399, 0.1661,
    #     0.1545, 0.1535, 0.1301, 0.1575, 0.1333, 0.1102, 0.1002, 0.1068, 0.1001,
    #     0.1165, 0.1014, 0.1165, 0.1014, 0.1490, 0.1305, 0.1490, 0.1305, 0.1447,
    #     0.1423, 0.1423, 0.1420, 0.1682, 0.1398, 0.1585, 0.1548, 0.1507, 0.1392,
    #     0.1543, 0.1306, 0.1052, 0.1002, 0.1017, 0.1002]
    # mean 
    # [0.2884, 0.2994, 0.2884, 0.2993, 0.0777, 0.1251, 0.0793, 0.1243, 0.1755,
    #     0.0888, 0.1761, 0.0880, 0.0589, 0.0272, 0.0604, 0.0258, 0.02chnl, 0.0144,
    #     0.0306, 0.0136, 0.0077, 0.0095, 0.0083, 0.0092, 0.0206, 0.0220, 0.0226,
    #     0.0208, 0.0240, 0.0251, 0.0269, 0.0234, 0.01chnl, 0.0199, 0.0223, 0.0183,
    #     0.0162, 0.0164, 0.0187, 0.0149, 0.0105, 0.0139, 0.0118, 0.0131, 0.0067,
    #     0.0095, 0.0073, 0.0091, 0.0205, 0.0216, 0.0212, 0.0210, 0.0145, 0.0185,
    #     0.0158, 0.0176, 0.0226, 0.0161, 0.0256, 0.0143, 0.0121, 0.0108, 0.0141,
    #     0.0098, 0.0078, 0.0087, 0.0089, 0.0081, 0.0049, 0.0060, 0.0053, 0.0057,
    #     0.0147, 0.0156, 0.0161, 0.0147, 0.0178, 0.0182, 0.0198, 0.0169, 0.0137,
    #     0.0139, 0.0156, 0.0126, 0.0114, 0.0109, 0.0132, 0.0098, 0.0072, 0.0087,
    #     0.0083, 0.0081, 0.0046, 0.0060, 0.0050, 0.0057]
    # stds
    # [0.1129, 0.1171, 0.1129, 0.1170, 0.0293, 0.0463, 0.02chnl, 0.0465, 0.0656,
    #     0.0315, 0.0651, 0.0314, 0.0219, 0.0198, 0.0226, 0.0183, 0.0126, 0.0221,
    #     0.0135, 0.0214, 0.0103, 0.0160, 0.0108, 0.0158, 0.0301, 0.0293, 0.0322,
    #     0.0278, 0.0368, 0.0339, 0.0397, 0.0317, 0.0278, 0.0268, 0.0309, 0.0244,
    #     0.0236, 0.0238, 0.0263, 0.0220, 0.0156, 0.0224, 0.0172, 0.0217, 0.0098,
    #     0.0161, 0.0102, 0.0158, 0.0271, 0.0282, 0.0279, 0.0275, 0.0176, 0.0219,
    #     0.0189, 0.0210, 0.0312, 0.0208, 0.0339, 0.0189, 0.0162, 0.0141, 0.0181,
    #     0.0130, 0.0103, 0.0124, 0.0115, 0.0119, 0.0066, 0.0087, 0.0071, 0.0085,
    #     0.0203, 0.0192, 0.0214, 0.0183, 0.0261, 0.0232, 0.0276, 0.0218, 0.0189,
    #     0.0174, 0.0205, 0.0159, 0.0160, 0.0143, 0.0177, 0.0131, 0.0102, 0.0124,
    #     0.0113, 0.0119, 0.0061, 0.0087, 0.0065, 0.0086]

    stds = torch.zeros(chnl).cuda() # 0.0705
    # pixs = 0
    mean = sums.reshape(chnl,1)/pixs
    # mean = torch.FloatTensor([0.2884, 0.2994, 0.2884, 0.2993, 0.0777, 0.1251, 0.0793, 0.1243, 0.1755,
    #     0.0888, 0.1761, 0.0880, 0.0589, 0.0272, 0.0604, 0.0258, 0.02chnl, 0.0144,
    #     0.0306, 0.0136, 0.0077, 0.0095, 0.0083, 0.0092, 0.0206, 0.0220, 0.0226,
    #     0.0208, 0.0240, 0.0251, 0.0269, 0.0234, 0.01chnl, 0.0199, 0.0223, 0.0183,
    #     0.0162, 0.0164, 0.0187, 0.0149, 0.0105, 0.0139, 0.0118, 0.0131, 0.0067,
    #     0.0095, 0.0073, 0.0091, 0.0205, 0.0216, 0.0212, 0.0210, 0.0145, 0.0185,
    #     0.0158, 0.0176, 0.0226, 0.0161, 0.0256, 0.0143, 0.0121, 0.0108, 0.0141,
    #     0.0098, 0.0078, 0.0087, 0.0089, 0.0081, 0.0049, 0.0060, 0.0053, 0.0057,
    #     0.0147, 0.0156, 0.0161, 0.0147, 0.0178, 0.0182, 0.0198, 0.0169, 0.0137,
    #     0.0139, 0.0156, 0.0126, 0.0114, 0.0109, 0.0132, 0.0098, 0.0072, 0.0087,
    #     0.0083, 0.0081, 0.0046, 0.0060, 0.0050, 0.0057]).reshape(chnl, 1).cuda()
    for i, data in enumerate(dataloader, 0):
        # pbar.update(int((i / (ll - 1)) * 100))
        X = Variable(data[0].cuda(), requires_grad=False)
        # Forward
        # gaborfeat = gaborext((X+1)/2).reshape(chnl,-1)
        gaborfeat = gaborext(X).reshape(chnl,-1)
        # pixs += gaborfeat.shape[1]
        stds += ((gaborfeat-mean)**2).sum(dim=1)
        gaborfeat = None
        torch.cuda.empty_cache()
    res = (stds/pixs).sqrt()
    print(res)
    torch.save(res, "featstd_6.pt")
    # pbar.finish()

# =============================== TRAINING ====================================
batch_size=args.batchSize
gaborext = Illust2vecNet('models/illust2vec_tag_ver200_2.pth')
gaborext.eval()

gaborext.cuda()
with torch.no_grad():
    test()
