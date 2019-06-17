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


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--zdim', type=int, default=3, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--start', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='gabor_vae2',  help='output folder')
parser.add_argument('--model', type=str, default='unet',  help='model type')
parser.add_argument('--initmodel', type=str, default='',  help='initialized model path')
parser.add_argument('--lr', type=float, default=0.0001,  help='learning rate')
parser.add_argument('--dataset', type=str, default='imgs',  help='dataset mode')
parser.add_argument('--cropSize', type=int, default=512, help='input batch size')
parser.add_argument('--lr_policy', type=str, default='step', help='input batch size')
parser.add_argument('--niter', type=int, default=200, help='input batch size')
parser.add_argument('--niter_decay', type=int, default=200, help='input batch size')
parser.add_argument('--preprocess', type=str, default='crop', help='input batch size')
parser.add_argument('--no_flip', action='store_true', help='input batch size')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
dataset = DataDataset('data/%s'%args.dataset, crop_size=args.cropSize)
# dataset = DataDataset('data/p11')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=args.workers)
tdataset = DataDataset('data/%s-test'%args.dataset, crop_size=args.cropSize)
# tdataset = DataDataset('data/p11-test')
test_loader = torch.utils.data.DataLoader(tdataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=args.workers)
size=args.cropSize
print(len(dataset))


def train(epoch, args):
    model.train()
    baseit = (epoch-1)*len(dataset)
    it = 0
    for i, data in enumerate(dataloader, 0):
        it+=batch_size
        X = Variable(data.cuda(), requires_grad=True)
        # Forward
        gaborfeat = gaborext((X+1)/2)
        _, _,_,_,intert = extractor(gaborfeat)
        interout = torch.clamp(intert, -1,1)
        inter, recons = model(X)
        solver.zero_grad()
        recons_loss = F.l1_loss(recons, X)
        inter_loss = F.l1_loss(inter, interout)
        loss = recons_loss + inter_loss
        loss.backward()
        solver.step()
        # print("epoch: %d iter: %d train_loss: %.04f gabor_loss: %.04f inter_loss: %.04f"%(epoch, it, loss.item(), gabor_loss.item(), inter_loss.item()))

        if it % print_interval == 0:
            train_writer.add_scalar('train_loss', loss.item(), baseit + it)
            train_writer.add_scalar('recons_loss', recons_loss.item(), baseit + it)
            train_writer.add_scalar('inter_loss', inter_loss.item(), baseit + it)
            print("epoch: %d iter: %d train_loss: %.04f recons_loss: %.04f inter_loss: %.04f"%
                (epoch, it, loss.item(), recons_loss.item(), inter_loss.item()))

    if epoch % save_interval == 0:
        print("Saving models...")
        torch.save(model.cpu().state_dict(), 'checkpoints2/%s/model_%d.pth' % (args.outf, epoch))
        model.cuda()
        torch.save(solver,  'checkpoints2/%s/optimizer_%d.pth' % (args.outf, epoch))

def test(epoch, args):
    model.eval()
    timg = next(iter(test_loader))
    X = Variable(timg.cuda(), requires_grad=False)
    # Forward
    gaborfeat = gaborext((X+1)/2)
    _, _,_,_,intert = extractor(gaborfeat)
    inter, recons = model(X)
    interout = torch.clamp(intert, -1,1)
    # Loss
    recons_loss = F.l1_loss(recons, X)
    inter_loss = F.l1_loss(inter, interout)
    loss = recons_loss + inter_loss
    train_writer.add_scalar('test_loss', loss.item(), epoch)
    train_writer.add_image('input', (timg+1)/2.0, epoch)
    train_writer.add_image('target', (interout.data+1)/2.0, epoch)
    train_writer.add_image('output', (inter.data+1)/2.0, epoch)


# =============================== TRAINING ====================================
print_interval = 400
save_interval = 10
batch_size=args.batchSize

try:
    os.mkdir('checkpoints2/%s'%args.outf)
except:
    pass

gaborext = GaborWavelet()
gaborext.eval()
extractor = GaborVAE()
extractor.load_state_dict(torch.load('checkpoints/gabor_step_i3_vae_mu/model_90.pth'))
# extractor.load_state_dict(torch.load('checkpoints/gabor_kl_step_1_i3_3c/model_500.pth'))
extractor.eval()

model = GaborAE()
# print(model)
if args.start>1:
    model.load_state_dict(torch.load('checkpoints2/%s/model_%d.pth' % (args.outf, args.start-1)))
    # solver.load_state_dict(torch.load('checkpoints2/%s/optimizer_%d.pth' % (args.outf, args.start-1)))
else:
    init_weights(model, init_type='kaiming')

solver = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5,0.9))#, weight_decay=1e-5)
scheduler = get_scheduler(solver, args)
train_writer = tensorboardX.SummaryWriter("./logs2/%s/"%args.outf)

gaborext.cuda()
extractor.cuda()
model.cuda()
for epoch in range(args.start, args.nepoch+1):
    train(epoch,args)
    with torch.no_grad():
        test(epoch,args)
    scheduler.step()
