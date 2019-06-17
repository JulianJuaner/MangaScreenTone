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
from networks1 import GaborVAE, GaborWavelet, init_weights,get_scheduler, relu


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--zdim', type=int, default=3, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--start', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='gabor_vae2',  help='output folder')
parser.add_argument('--model', type=str, default='unet',  help='model type')
parser.add_argument('--initmodel', type=str, default='',  help='initialized model path')
parser.add_argument('--lr', type=float, default=0.00005,  help='learning rate')
parser.add_argument('--dataset', type=str, default='imgs',  help='dataset mode')
parser.add_argument('--cropSize', type=int, default=512, help='input batch size')
parser.add_argument('--lr_policy', type=str, default='step', help='input batch size')
parser.add_argument('--niter', type=int, default=200, help='input batch size')
parser.add_argument('--niter_decay', type=int, default=200, help='input batch size')
parser.add_argument('--preprocess', type=str, default='crop', help='input batch size')
parser.add_argument('--no_flip', action='store_true', help='input batch size')
parser.add_argument('--sqr', type=int, default=0, help='input batch size')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
dataset = DataDataset('../cv2/inpainting/data/screentone/', crop_size=args.cropSize)
# dataset = DataDataset('data/p11')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=args.workers)
tdataset = DataDataset('../cv2/inpainting/data/test/screentone/', crop_size=args.cropSize)
# tdataset = DataDataset('data/p11-test')
test_loader = torch.utils.data.DataLoader(tdataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=args.workers)
size=args.cropSize
print(len(dataset))


def train(epoch, args):
    model.train()
    lamb = (1. / (1. + math.exp(- 0.1 * epoch + 5)))*0.2#+0.05
    print(lamb)
    baseit = (epoch-1)*len(dataset)
    it = 0
    for i, data in enumerate(dataloader, 0):
        it+=batch_size
        X = Variable(data.cuda(), requires_grad=True)
        # Forward
        gaborfeat = gaborext((X+1)/2)
        gaborfeat.detach_() 
        inter, reconsgabor, reconsinter, logvar, mu = model(gaborfeat)
        # gabor_loss = recon_loss(reconsgabor, gaborfeat)#/batch_size
        # inter_loss = recon_loss(reconsinter, inter)#/batch_size
        solver.zero_grad()
        gabor_loss = F.l1_loss(reconsgabor, gaborfeat)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) - 1. - logvar, 1))
        if args.sqr == 1:
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, 1))
        #kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu882 -1 - logvar, 1))
        loss = gabor_loss + kl_loss*lamb
        loss.backward(retain_graph=True)
        # solver.step()
        # solver.zero_grad()
        inter.detach_()
        #inter_loss = F.l1_loss(reconsinter, inter)#*10
        # inter_loss_bw = inter_loss*lamb*10
        # inter_loss_bw.backward()
        # loss += inter_loss_bw
        #inter_loss.backward()
        #loss += inter_loss
        solver.step()
        # gabor_loss = torch.abs(reconsgabor-gaborfeat).mean()
        # inter_loss = torch.abs(reconsinter-inter).mean()
        # print("epoch: %d iter: %d train_loss: %.04f gabor_loss: %.04f inter_loss: %.04f"%(epoch, it, loss.item(), gabor_loss.item(), inter_loss.item()))

        if it % print_interval == 0:
            train_writer.add_scalar('train_loss', loss.item(), baseit + it)
            train_writer.add_scalar('gabor_loss', gabor_loss.item(), baseit + it)
            #train_writer.add_scalar('inter_loss', inter_loss.item(), baseit + it)
            train_writer.add_scalar('kl_loss', kl_loss.item(), baseit + it)
            print("epoch: %d iter: %d train_loss: %.04f gabor_loss: %.04f  kl_loss: %.04f"%
                (epoch, it, loss.item(), gabor_loss.item(), kl_loss.item()))

    # Print and plot every now and then
    #print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))
    if epoch % save_interval == 0:
        print("Saving models...")
        torch.save(model.cpu().state_dict(), 'checkpoints/%s/model_%d.pth' % (args.outf, epoch))
        model.cuda()
        torch.save(solver,  'checkpoints/%s/optimizer_%d.pth' % (args.outf, epoch))

def test(epoch, args):
    model.eval()
    timg = next(iter(test_loader))
    X = Variable(timg.cuda(), requires_grad=False)
    # Forward
    gaborfeat = gaborext((X+1)/2)
    inter, reconsgabor, reconsinter, logvar, mu = model(gaborfeat)
    # Loss
    # gabor_loss = recon_loss(reconsgabor, gaborfeat)#/batch_size
    # inter_loss = recon_loss(reconsinter, inter)#/batch_size
    loss = F.l1_loss(reconsgabor, gaborfeat)
    #inter_loss = F.l1_loss(reconsinter, inter)
    # gabor_loss = torch.abs(reconsgabor-gaborfeat).mean()
    # inter_loss = torch.abs(reconsinter-inter).mean()
    #kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + (mu+1) - 1. - logvar, 1))
    #loss = gabor_loss + inter_loss
    train_writer.add_scalar('test_loss', loss.item(), epoch)
    train_writer.add_image('input', (timg[1]+1)/2.0, epoch)
    # train_writer.add_image('output', (inter[:,:1,:,:].data+1)/2.0, epoch)
    # print(inter[:,:1,:,:].data.min().item(),inter[:,:1,:,:].data.max().item())
    # print(inter[:,1:,:,:].data.min().item(),inter[:,1:,:,:].data.max().item())
    # train_writer.add_image('output', (inter[:,:3,:,:].data+1)/2.0, epoch)
    interout = torch.clamp((inter[:,:3,:,:].data+1)/2.0, 0,1)
    train_writer.add_image('output', interout[1], epoch)
    print(inter[:,:3,:,:].data.min().item(),inter[:,:3,:,:].data.max().item())
    print(inter[:,3:,:,:].data.min().item(),inter[:,3:,:,:].data.max().item())
    # f=open('testit2.txt', 'w+')
    # # f.writelines(["%s " % item  for item in mu])
    # f.writelines(["%s " % item  for item in torch.clamp(mu.cpu().data,-1,1)])


# =============================== TRAINING ====================================

print_interval = 50
save_interval = 3
batch_size=args.batchSize
gaborext = GaborWavelet()
gaborext.eval()
model = GaborVAE()

try:
    os.mkdir('checkpoints/%s'%args.outf)
except:
    pass

solver = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-3)
scheduler = get_scheduler(solver, args)
train_writer = tensorboardX.SummaryWriter("./logs/%s/"%args.outf)

# print(model)
if args.start>1:
    model.load_state_dict(torch.load('checkpoints/%s/model_%d.pth' % (args.outf, args.start-1)))
    # solver.load_state_dict(torch.load('checkpoints/%s/optimizer_%d.pth' % (args.outf, args.start-1)))
else:
    init_weights(model, init_type='kaiming')

# recon_loss = nn.L1Loss()#nn.BCELoss() reduction='elementwise_mean'

gaborext.cuda()
model.cuda()
for epoch in range(args.start, args.nepoch+1):
    train(epoch,args)
    with torch.no_grad():
        test(epoch,args)
    scheduler.step()
