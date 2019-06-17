from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.autograd import Variable
from datasets1 import DataDataset
import math
import tensorboardX
from networks1 import GaborVAE, GaborWavelet, init_weights,get_scheduler, GaborAE
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--zdim', type=int, default=3, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--start', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='gabor_kl_step_1_i3',  help='output folder')
parser.add_argument('--model', type=str, default='resnet',  help='model type')
parser.add_argument('--initmodel', type=str, default='',  help='initialized model path')
parser.add_argument('--lr', type=float, default=0.0002,  help='learning rate')
parser.add_argument('--dataset', type=str, default='imgs',  help='dataset mode')
parser.add_argument('--cropSize', type=int, default=1936, help='input batch size')
parser.add_argument('--lr_policy', type=str, default='step', help='input batch size')
parser.add_argument('--niter', type=int, default=200, help='input batch size')
parser.add_argument('--niter_decay', type=int, default=200, help='input batch size')
parser.add_argument('--preprocess', type=str, default='none', help='input batch size')
parser.add_argument('--no_flip', action='store_true', help='input batch size')
args = parser.parse_args()

tdataset = DataDataset('data/%s'%args.dataset, crop_size=args.cropSize, train=False)
# tdataset = DataDataset('data/p11-test')
test_loader = torch.utils.data.DataLoader(tdataset, batch_size=1,
                                         shuffle=False, num_workers=args.workers)
size=args.cropSize


def test(args):
    for i, data in enumerate(test_loader, 0):
        # print(data[1])
        X = Variable(data[0].cuda(), requires_grad=False)
        # Forward
        mu = model(X)
        # gaborfeat = gaborext((X+1)/2)
        # inter, reconsgabor, reconsinter, logvar, mu = model(gaborfeat)
        # out = torch.clamp((mu[0]+1)*127.5,0,255)
        out = (mu[0]+1)*127.5
        cv2.imwrite(data[1][0].replace('data/','results/').replace('.jpg','.png'), cv2.cvtColor(out.cpu().numpy().transpose(1,2,0), cv2.COLOR_RGB2BGR))
        # out = torch.clamp(inter[0]*200+200,0,400)*255//400
        # cv2.imwrite('results/%s/%d-in.png'%(args.outf,i), (data[0,0].cpu().numpy()+1)*127.5)
        # cv2.imwrite('results/%s/%d-out.png'%(args.outf,i), cv2.cvtColor(out[:3,:,:].cpu().numpy().transpose(1,2,0), cv2.COLOR_RGB2BGR))



batch_size=args.batchSize
# gaborext = GaborWavelet()
# gaborext.eval()
model = GaborAE()
model.eval()

try:
    os.mkdir('results/%s'%args.outf)
except:
    pass

# print(model)
if args.start>1:
    model.load_state_dict(torch.load('checkpoints2/%s/model_%d.pth' % (args.outf, args.start)))
    # solver.load_state_dict(torch.load('checkpoints/%s/optimizer_%d.pth' % (args.outf, args.start-1)))
else:
    init_weights(model, init_type='xavier')

# gaborext.cuda()
model.cuda()
with torch.no_grad():
    test(args)
