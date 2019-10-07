from __future__ import print_function
import sys
sys.path.append('..')
from models.networks import Illust2vecNet, MultiLayer
from offline import two_addition
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import cv2
import numpy as np
import torchvision
from torch.autograd import Variable

import math
from datasets import *
import argparse
import tensorboardX
import time
from networks1 import init_weights, get_scheduler
from encoder import FeatureEncoder, VAE, FeatureMap

CHANNEL = 64
name = 'simline'
w = 512
h = 512
parser = argparse.ArgumentParser()
is_vae = False
triplet = False

parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--zdim', type=int, default=3, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--start', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='32_Triplet_relative', help='output folder')
parser.add_argument('--model', type=str, default='unet',  help='model type')
parser.add_argument('--initmodel', type=str, default='',  help='initialized model path')
parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
parser.add_argument('--dataset', type=str, default='../../../../data/manga',  help='dataset mode')
parser.add_argument('--cropSize', type=int, default=512, help='input batch size')
parser.add_argument('--lr_policy', type=str, default='step', help='input batch size')
parser.add_argument('--niter', type=int, default=200, help='input batch size')
parser.add_argument('--niter_decay', type=int, default=200, help='input batch size')
parser.add_argument('--preprocess', type=str, default='crop', help='input batch size')
parser.add_argument('--no_flip', action='store_true', help='input batch size')
parser.add_argument('--sqr', type=int, default=0, help='square kl loss')
parser.add_argument('--mode', type=str, default='train')
opt = parser.parse_args()
print(opt)
manga_pth = opt.dataset
print_interval = 300

trainloaderA = DataDataset(manga_pth, base=0, name=name, mode='feature_compress')
trainloader = torch.utils.data.DataLoader(trainloaderA, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=opt.workers)
#distanceEval = torch.nn.L2loss()
if is_vae:
    model = VAE().cuda()
else:
    model = FeatureEncoder().cuda()
i2v = MultiLayer('../models/illust2vec_tag_ver200_2.pth').eval().cuda()

distance = torch.nn.MSELoss(reduce=False).cuda()
dist = torch.nn.SmoothL1Loss().cuda()
compare = torch.nn.L1Loss().cuda()
KL = torch.nn.KLDivLoss().cuda()

if triplet:
    triloss = torch.nn.TripletMarginLoss().cuda()

#trainloaderB = DataDataset(manga_pth, base=0, name=name, mode='feature')

def train(epoch, opt):
    lamb = (1. / (1. + math.exp(- 0.1 * (epoch) + 5)))*0.2#+0.05
    print(lamb)
    model.train()
    baseit = (epoch-1)*len(trainloaderA)
    it = 0-opt.batchSize
    #NormalDistribution = torch.FloatTensor()
    for i, data in enumerate(trainloader):
        it+=opt.batchSize
        X = i2v(Variable(data[0].cuda(), requires_grad=True))
        Y = i2v(Variable(data[1].cuda(), requires_grad=True))
        if triplet:
            Z = Variable(data[2].cuda(), requires_grad=True)
            Xp = model(X)
            Yp = model(Y)
            Zp = model(Z)

            disxy = dist(X, Y)
            disyz = dist(Y, Z)
            disxz = dist(X, Z)

            if disxy.item()<=disxz.item() and disxz.item()<=disyz.item():
                loss = compare(triloss(Yp, Xp, Zp), triloss(Y, X, Z)) #a p n
            elif disxy.item()<=disyz.item() and disyz.item()<=disxz.item():
                loss = compare(triloss(Xp, Yp, Zp), triloss(X, Y, Z))
            elif disxz.item()<=disxy.item() and disxy.item()<=disyz.item():
                loss = compare(triloss(Zp, Xp, Yp), triloss(Z, X, Y))
            elif disxz.item()<=disyz.item() and disyz.item()<=disxy.item():
                loss = compare(triloss(Xp, Zp, Yp), triloss(X, Z, Y))
            elif disyz.item()<=disxy.item() and disxy.item()<=disxz.item():
                loss = compare(triloss(Zp, Yp, Xp), triloss(Z, Y, X))
            elif disyz.item()<=disxz.item() and disxz.item()<=disxy.item():
                loss = compare(triloss(Yp, Zp, Xp), triloss(Y, Z, X))
            else:
                print("wrong!")
            loss.backward(retain_graph=True)
            solver.step()

            if it % print_interval == 0:
                #print(loss.item(), D2.item())
                print("epoch: %d iter: %d train_loss: %.07f"%
                    (epoch, it, loss.item()))
                train_writer.add_scalar('loss', loss.item(), baseit + it)
                train_writer.add_image('output', torchvision.utils.make_grid(F.normalize(Xp[:, :3, :, :]), baseit + it))
            
        elif is_vae == True:
            x_inter, x_reconsgabor, x_reconsinter, x_logvar, x_mu, repreX = model(X)
            y_inter, y_reconsgabor, y_reconsinter, y_logvar, y_mu, repreY = model(Y)
            solver.zero_grad()
            #x_gabor_loss = compare(x_reconsgabor, X)
            #y_gabor_loss = compare(y_reconsgabor, Y)
            D1 = dist(X,Y)
            D2 = dist(x_mu, y_mu)
            
            #D1 = torch.mean(distance(X, Y), dim=1, keepdim=True)
            #D2 = torch.mean(distance(x_reconsinter, y_reconsinter), dim=1, keepdim=True)
            #D2 = distance(x_reconsinter, y_reconsinter)
            diff = compare(D1, D2)
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(x_logvar) - 1. + x_mu**2 - x_logvar, 1))
            #inter_loss = compare(x_inter, x_reconsinter)
            loss = kl_loss*lamb + diff #+ x_gabor_loss + inter_loss
            
            loss.backward(retain_graph=True)
            solver.step()

            if it % print_interval == 0:
                print(D1.item(), D2.item())
                print("epoch: %d iter: %d train_loss: %.07f, kl: %.07f"%
                    (epoch, it, diff.item(), kl_loss.item()))
                train_writer.add_scalar('loss', loss.item(), baseit + it)
                train_writer.add_scalar('kl_loss', kl_loss.item(), baseit + it)
                train_writer.add_scalar('diff', diff.item(), baseit + it)
                #train_writer.add_scalar('gabor', x_gabor_loss.item(), baseit + it)
                train_writer.add_image('output', torchvision.utils.make_grid(F.normalize(x_inter[:,:3,:,:])), baseit + it) 
                train_writer.add_image('original_img', torchvision.utils.make_grid(data[0][:,:3,:,:]), baseit + it)    

        else:
            #Z = i2v(Variable(data[2].cuda(), requires_grad=True))
            solver.zero_grad()
            #Xp = torch.FloatTensor(two_addition(X[0].detach().cpu().numpy())).cuda().unsqueeze(0)
            #Yp = torch.FloatTensor(two_addition(Y[0].detach().cpu().numpy())).cuda().unsqueeze(0)
            Yp = model(Y)
            Xp = model(X)
            #Zp = model(Z)
            # Implementation of feature map loss.
            FeatureMap_Before = FeatureMap(Y.detach(), 2048).cuda()
            FeatureMap_After = FeatureMap(Yp.detach(), 2048).cuda()
            Map_before = FeatureMap_Before(X)
            Map_after = FeatureMap_After(Xp)
            
            #diff scale implementation.
            #FeatureMap
            loss = dist(torch.pow(Map_before+0.5, 2), torch.pow(Map_after+0.5, 2)) #+ (actual_diff + expect_diff + value_diff)*60*(Map_before[0][0][max_idyB.item()][max_idxB.item()].item()-0.25)
        
            loss.backward(retain_graph=True)
            solver.step()
            if it % print_interval == 0:
                #print(D1.item(), D2.item())
                print("%s: epoch: %d iter: %d train_loss: %.07f,"%# idx_loss: %.07f, best_loss: %.07f"%
                    (opt.outf, epoch, it, loss.item()))
                train_writer.add_scalar('loss', loss.item(), baseit + it)
                print(X.shape, Y.shape)
                train_writer.add_image('kernel', torchvision.utils.make_grid(data[1][:,:3,:,:]/255), baseit + it)
                train_writer.add_image('output', torchvision.utils.make_grid(F.normalize(Xp[:, :3, :, :])), baseit + it)
                train_writer.add_image('feature_map', torchvision.utils.make_grid(torch.clamp((Map_before), 0,1)), baseit + it)
                train_writer.add_image('feature_map_after', torchvision.utils.make_grid(torch.clamp((Map_after), 0,1)), baseit + it)
                train_writer.add_image('original_img', torchvision.utils.make_grid(data[0][:,:3,:,:]/255), baseit + it)

    if epoch % save_interval == 0:
        print("Saving models...")
        torch.save(model.cpu().state_dict(), 'checkpoints/%s/model_%d.pth' % (opt.outf, epoch))
        model.cuda()
        torch.save(solver, 'checkpoints/%s/optimizer_%d.pth' % (opt.outf, epoch))

print_interval = 100
save_interval = 1
batch_size=opt.batchSize

######TESTCASE#######
#testLoaderA = DataDataset(manga_pth, base=0, name=name, mode='feature')
#testLoader = torch.utils.data.DataLoader(testLoaderA, batch_size=opt.batchSize,
#                                         shuffle=True, num_workers=opt.workers)

NAME = 'image'
def testAndSave(opt):
    books = os.listdir(os.path.join(manga_pth,NAME))
    books.sort()
    name = NAME
    start = time.time()

    for book in books:
        try:
            os.makedirs(os.path.join(manga_pth, opt.outf, book), exist_ok=True)
        except:
            pass
    testLoaderA = DataDataset(manga_pth, base=0, name=name, mode='img', length=-1)
    testLoader = torch.utils.data.DataLoader(testLoaderA, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=opt.workers)

    for i, data in enumerate(testLoader):
        #it+=opt.batchSize
        X = i2v(Variable(data.cuda(), requires_grad=True))
        if is_vae:
            x_inter, x_reconsgabor, x_reconsinter, x_logvar, x_mu, repreX = model(X)
            Xp = x_mu.cpu().detach().numpy()
        else:
            if i%100==0:
                print(i)
            #Xp = torch.FloatTensor(two_addition(X[0].detach().cpu().numpy())).cuda().unsqueeze(0)
            Xp = model(X)
            Xp = Xp.cpu().detach()
        for j in range(batch_size):
            print(os.path.join(opt.dataset,
            opt.outf,testLoaderA.data[i*batch_size+j].split('/')[-2],
            testLoaderA.data[i*batch_size+j].split('/')[-1].replace('png', 'pt')))

            if '.JPG' in testLoaderA.data[i*batch_size+j].split('/')[-1]:
                 torch.save(Xp[j], os.path.join(opt.dataset,
                opt.outf,testLoaderA.data[i*batch_size+j].split('/')[-2],
                testLoaderA.data[i*batch_size+j].split('/')[-1].replace('.JPG', '.pt')))
            else:
                torch.save(Xp[j], os.path.join(opt.dataset,
                opt.outf,testLoaderA.data[i*batch_size+j].split('/')[-2],
                testLoaderA.data[i*batch_size+j].split('/')[-1].replace('png', 'pt')))

        


try:
    os.mkdir('checkpoints/%s'%opt.outf)
except:
    pass

solver = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-8)
scheduler = scheduler = lr_scheduler.StepLR(solver, step_size=opt.niter_decay, gamma=0.1)


# print(model)
if opt.start>1:
    model.load_state_dict(torch.load('checkpoints/%s/model_%d.pth' % (opt.outf, opt.start-1)))
    #solver.load_state_dict(torch.load('checkpoints/%s/optimizer_%d.pth' % (opt.outf, opt.start-1)))
else:
    init_weights(model, init_type='kaiming')

# recon_loss = nn.L1Loss()#nn.BCELoss() reduction='elementwise_mean'

if opt.mode == 'train':
    for epoch in range(opt.start, opt.nepoch+1):
        train_writer = tensorboardX.SummaryWriter("./log/%s/"%opt.outf)
        train(epoch,opt)
        #with torch.no_grad():
        #    test(epoch,opt)
        scheduler.step()
else:
    testAndSave(opt)
