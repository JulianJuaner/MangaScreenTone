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

    sums = torch.zeros(chnl).cuda() # 0.0358
    mins = torch.ones(chnl).cuda() # 0
    maxs = torch.zeros(chnl).cuda() # 0.4022
    pixs = 0

    for i, data in enumerate(dataloader, 0):

        X = Variable(data[0].cuda(), requires_grad=False)

        gaborfeat = gaborext(X).reshape(chnl,-1)
        pixs += gaborfeat.shape[1]
        sums += gaborfeat.sum(dim=1)

        gaborfeat = None
        torch.cuda.empty_cache()

    print(sums/pixs)
    torch.save(sums/pixs, "featmean_4.pt")


    stds = torch.zeros(chnl).cuda() # 0.0705
    # pixs = 0
    mean = sums.reshape(chnl,1)/pixs
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
    torch.save(res, "featstd_4.pt")
    # pbar.finish()

# =============================== TRAINING ====================================
batch_size=args.batchSize
gaborext = Illust2vecNet('models/illust2vec_tag_ver200_2.pth')
gaborext.eval()

gaborext.cuda()
with torch.no_grad():
    test()
