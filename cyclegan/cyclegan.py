import argparse
import os
import numpy as np
import math
import itertools
import datetime
import collections
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from perceptual import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import tensorboardX

parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', type=str, default='train')
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="manga", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--lambda_mask", type=float, default=1.0, help="mask loss weight")
parser.add_argument("--lambda_perc", type=float, default=1.0, help="perceptual loss weight")
parser.add_argument("--datasize", type=int, default=13000, help="size of a dataset")
parser.add_argument("--outf", type=str, default="manga01_mask_error", help="output folder")
parser.add_argument("--loss_mode", type=str, default="image", help="The error mode of the image")
parser.add_argument("--stage", type=int, default=-1, help="Stage for perceptual loss")
parser.add_argument("--test_size", type=int, default=5, help="size for valid loader")

#parser = argparse.ArgumentParser()
#dataset settings.
parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true',default=False, help='if specified, do not flip the images for data augmentation')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
parser.add_argument('--num_img', type=int, default=6, help='# of imgaes in set A')

opt = parser.parse_args()
from datasets import *
print(opt)
class CYCLEGAN:
    def __init__(self, opt):
        self.train_writer = tensorboardX.SummaryWriter("./logs/%s/"%opt.outf)

        #loss name dict.
        self.loss_names = ['GAN_AB', 'GAN_BA', 'cycle_A', 'cycle_B', 'id_A', 'id_B',\
                    'D_A', 'D_B', 'perc_A', 'perc_B','G','D','cycle','identity','line']
        if opt.loss_mode == 'mask':
            self.loss_names = ['GAN_AB', 'GAN_BA', 'cycle_A', 'cycle_B', 'id_A', 'id_B', 'mask_A',\
                     'mask_B', 'D_A', 'D_B', 'perc_A', 'perc_B','G','D','cycle','mask_','identity']

        cuda = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

        # Training data loader
        self.dataloader = DataLoader(
            ImageDataset("../../data/%s" % opt.dataset_name, \
            "../../data/%s" % opt.dataset_name, \
            channel = opt.channels, datasize = opt.datasize, transforms_=None, unaligned=False, mask=True, opt = opt),
            batch_size=opt.batch_size, 
            shuffle=True,
            num_workers=0,
        )
        # Test data loader
        self.val_dataloader = DataLoader(
            ImageDataset("../../data/%s" % opt.dataset_name, \
            "../../data/%s" % opt.dataset_name, \
            channel = opt.channels, datasize = 200, transforms_=None, unaligned=False, mode="test", mask=True, opt = opt),
            batch_size=opt.test_size,
            shuffle=True,
            num_workers=0,
        )


    def sample_images(self, batches_done, opt):
        """Saves a generated sample from the test set"""
        imgs = next(iter(self.val_dataloader))
        self.G_AB.eval()
        self.G_BA.eval()
        real_A = Variable(imgs["A"].type(self.Tensor))
        fake_B = self.G_AB(real_A)
        real_B = Variable(imgs["B"].type(self.Tensor))
        fake_A = self.G_BA(real_B)

        real_A = make_grid(real_A, nrow=5, padding=0, normalize=False)
        real_B = make_grid(real_B, nrow=5, padding=0, normalize=False)
        fake_A = make_grid(fake_A, nrow=5, padding=0, normalize=False)
        fake_B = make_grid(fake_B, nrow=5, padding=0, normalize=False)

        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        R = image_grid.cpu().detach().numpy()
        
        cv2.imwrite("images/%s/%s_%s.jpg" % (opt.outf, opt.outf, batches_done), R.transpose(1,2,0))
        self.train_writer.add_image('test result', image_grid, batches_done)

    '''PART ADDED'''

    def loss_mask(img, mode, mask_img, opt):
        if mode=='image':
            return img
        elif mode == 'mask':
            mask = mask_img.cpu().detach().numpy().transpose(0,2,3,1)
            for layer in range(len(mask)):
                mask[layer,:,:,0] = Erosion(np.uint8(mask[layer]*255), 7).astype(float)/255
                
            mask = torch.FloatTensor(mask.transpose(0,3,1,2))
            #SaveFirstImage(mask, '180test.png')
            
            for ch in range(opt.channels):
                img[:,ch,:,:].unsqueeze(1)[mask<0.5] = 1
            #SaveFirstImage(img, '180image.png')
            return img

    def get_current_losses(self):
            """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
            errors_ret = collections.OrderedDict()
            for name in self.loss_names:
                if isinstance(name, str):
                    errors_ret[name] = float(getattr(self, 'loss_' + name).item())  # float(...) works for both scalar tensor and float number
            return errors_ret

    def applymask(self, image, mask, mode='real'):
        mask_img=mask.clone().cpu().detach().numpy().transpose(0,2,3,1)
        img = image.clone()
        for layer in range(len(mask_img)):
            mask_img[layer,:,:,0] = Threshold(np.uint8(mask_img[layer]*255)).astype(float)/255
        mask_img = torch.FloatTensor(mask_img.transpose(0,3,1,2))
        if mode =='real':
            return mask_img.cuda()
        #SaveFirstImage(mask_img, 'mask_img.png')
        for ch in range(opt.channels):
            img[:,ch,:,:].unsqueeze(1)[mask_img>0.5] = 1
        #SaveFirstImage(img, 'imgg.png')
        return img.cuda()
    '''END'''
    # ----------
    # Testing
    # ----------
    def test(self, opt, folder = '../../data/manga', sub='sim'):
        books = os.listdir(os.path.join(folder, 'simline'))
        books.sort()
        print(os.path.join(folder, sub))
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
        start = time.time()
        print('start')
        print(books)
        for book in books:
            try:
                os.mkdir(os.path.join(folder, sub, book))
                subbooks = os.listdir(os.path.join(folder, 'image', book))
                print(subbooks)
                for subbook in subbooks:
                    try:
                        print(os.path.join(folder, sub, book, subbook))
                        if os.path.isdir(os.path.join(folder, 'image', book, subbook)):
                            os.mkdir(os.path.join(folder, sub, book, subbook))
                            #print(os.path.join(folder, sub, book, subbook))
                    except:
                        print('no such book')
                        pass

            except:
                pass

        print("making dataset...")
        dataset = TestDataset(os.path.join(folder, 'image'), opt)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                shuffle=False, num_workers=1)  
        input_shape = (opt.channels, (1170//4)*4, (827//4)*4)
        print("start")
        self.G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
        self.G_BA.cuda()
        self.G_BA=nn.DataParallel(self.G_BA)
        self.G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.outf, opt.epoch)))
        self.G_BA.eval()
        already = 0
        for i,data in enumerate(dataloader):
            if i<already:
                continue
            img = Variable(data["Image"].type(self.Tensor))
            print(img.shape)
            if img.shape[2]>1600 or img.shape[3]>1600:
                if img.shape[3]<2200 and img.shape[2]<2400:
                    #img = img.detach().numpy().transpose(0,2,3,1)
                    img1 = img.clone()[:,:,:img.shape[2]//2, :]
                    img2 = img.clone()[:,:,img.shape[2]//2:, :]
                    result1 = self.G_BA(img1.cuda()).cpu().detach().numpy().transpose(0,2,3,1)[0]
                    result2 = self.G_BA(img2.cuda()).cpu().detach().numpy().transpose(0,2,3,1)[0]
                    result = np.concatenate((result1, result2), axis=0)
                else:
                    print("4")
                    base = img.shape[2]//4
                    img1 = img.clone()[:,:,:base, :]
                    img2 = img.clone()[:,:,base:2*base, :]
                    img3 = img.clone()[:,:,2*base:3*base, :]
                    img4 = img.clone()[:,:,3*base:, :]
                    result1 = self.G_BA(img1.cuda()).cpu().detach().numpy().transpose(0,2,3,1)[0]
                    result2 = self.G_BA(img2.cuda()).cpu().detach().numpy().transpose(0,2,3,1)[0]
                    result3 = self.G_BA(img3.cuda()).cpu().detach().numpy().transpose(0,2,3,1)[0]
                    result4 = self.G_BA(img4.cuda()).cpu().detach().numpy().transpose(0,2,3,1)[0]
                    result = np.concatenate((result1, result2, result3, result4), axis=0)
                print(result.shape)
            else:
                result = self.G_BA(img.cuda()).cpu().detach().numpy().transpose(0,2,3,1)[0]
            
            print(os.path.join(folder,
                sub,dataset.file_list[i].split('/')[-2],
                dataset.file_list[i].split('/')[-1]))
            print(dataset.file_list[i])
            if dataset.file_list[i].split('/')[-3] =='new':
                cv2.imwrite(os.path.join(folder,
                sub,dataset.file_list[i].split('/')[-3],dataset.file_list[i].split('/')[-2],
                dataset.file_list[i].split('/')[-1]),
                np.uint8(result*255))
            else:
                cv2.imwrite(os.path.join(folder,
                sub,dataset.file_list[i].split('/')[-2],
                dataset.file_list[i].split('/')[-1]),
                np.uint8(result*255))
            sys.stdout.write(
                "\r [Batch %d/%d]"
                % (
                    i,
                    len(dataloader),
                )
            )

            

    # ----------
    # Getting Results
    # ----------
    def get_result(self,opt):
        os.makedirs("images/%s/B/" % opt.outf, exist_ok=True)
        os.makedirs("images/%s/A/" % opt.outf, exist_ok=True)
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
        self.G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
        cuda = torch.cuda.is_available()
        device = torch.device("cuda")
        if cuda:
            self.G_AB = self.G_AB.cuda()
            self.G_AB=nn.DataParallel(self.G_AB)
            self.G_BA = self.G_BA.cuda()
            self.G_BA=nn.DataParallel(self.G_BA)
        if opt.epoch != 0:
            # Load pretrained models
            self.G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.outf, opt.epoch)))
            self.G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.outf, opt.epoch)))

        for index in range(len(self.dataloader)):
            """Saves a generated sample from the test set"""
            imgs = next(iter(self.dataloader))
            self.G_AB.eval()
            self.G_BA.eval()
            real_A = Variable(imgs["A"].type(self.Tensor))
            fake_B = self.G_AB(real_A)
            real_B = Variable(imgs["B"].type(self.Tensor))
            fake_A = self.G_BA(real_B)

            # Arange images along y-axis
            RA = fake_A.cpu().detach().numpy().transpose(0,2,3,1)

            for i in range(opt.batch_size):
                cv2.imwrite("images/%s/A/%s_%03d%d.jpg" % (opt.outf, opt.outf, index, i), 255*RA[i])

            sys.stdout.write(
                "\r [Batch %d/%d]"
                % (
                    index,
                    len(self.dataloader),
                )
            )
        pass

    # ----------
    #  Training
    # ----------
    def train(self, opt):
        white = torch.FloatTensor(np.ones((opt.batch_size, opt.channels, opt.img_height, opt.img_width)))
        # Create sample and checkpoint directories
        os.makedirs("images/%s" % opt.outf, exist_ok=True)
        os.makedirs("saved_models/%s" % opt.outf, exist_ok=True)

        # Losses
        criterion_GAN = torch.nn.MSELoss()
        criterion_cycle = torch.nn.L1Loss()
        #new loss: line loss... try first.
        criterion_line = torch.nn.L1Loss()

        criterion_identity = torch.nn.L1Loss()
        criterion_mask = torch.nn.L1Loss(reduction='sum')
        criterion_per = PercLoss(stage=opt.stage)

        cuda = torch.cuda.is_available()

        input_shape = (opt.channels, opt.img_height, opt.img_width)

        # Initialize generator and discriminator
        self.G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
        self.G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
        D_A = Discriminator(input_shape)
        D_B = Discriminator(input_shape)

        if cuda:
            white = white.cuda()

            criterion_per = criterion_per.cuda()

            self.G_AB = self.G_AB.cuda()
            self.G_AB=nn.DataParallel(self.G_AB)
            self.G_BA = self.G_BA.cuda()
            self.G_BA=nn.DataParallel(self.G_BA)
            D_A = D_A.cuda()
            DA_outshape = D_A.output_shape
            D_A=nn.DataParallel(D_A)
            D_B = D_B.cuda()
            DB_outshape = D_B.output_shape
            D_B=nn.DataParallel(D_B)

            criterion_GAN.cuda()
            criterion_cycle.cuda()
            criterion_line.cuda()
            criterion_identity.cuda()
            criterion_mask.cuda()

        if opt.epoch != 0:
            # Load pretrained models
            self.G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.outf, opt.epoch)))
            self.G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.outf, opt.epoch)))
            D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.outf, opt.epoch)))
            D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.outf, opt.epoch)))
        else:
            # Initialize weights
            self.G_AB.apply(weights_init_normal)
            self.G_BA.apply(weights_init_normal)
            D_A.apply(weights_init_normal)
            D_B.apply(weights_init_normal)

        # Optimizers
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
        )
        optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        # Learning rate update schedulers
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )
        # Buffers of previously generated samples
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()
        device = torch.device("cuda")
        prev_time = time.time()
        
        for epoch in range(opt.epoch+1, opt.n_epochs):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(batch["A"].type(self.Tensor))
                real_B = Variable(batch["B"].type(self.Tensor))
                if opt.loss_mode == 'mask':
                    A_mask = Variable(batch["maskA"].type(self.Tensor))
                    #print(A_mask.shape)
                    B_mask = Variable(batch["maskB"].type(self.Tensor))

                # Adversarial ground truths
                valid = Variable(self.Tensor(np.ones((real_A.size(0), *DA_outshape))), requires_grad=False)
                fake = Variable(self.Tensor(np.zeros((real_A.size(0), *DA_outshape))), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                self.G_AB.train()
                self.G_BA.train()

                optimizer_G.zero_grad()


                # Identity loss
                self.loss_identity = 0

                if opt.lambda_id!=-1:
                    self.loss_id_A = criterion_identity(self.G_AB(real_A), real_A)
                    self.loss_id_B = criterion_identity(self.G_BA(real_B), real_B)

                    self.loss_identity = (self.loss_id_A + self.loss_id_B) / 2

                # GAN loss
                fake_B = self.G_AB(real_A)
                fake_A = self.G_BA(real_B)

                self.loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                self.loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
                
                self.loss_GAN = (self.loss_GAN_AB + self.loss_GAN_BA)/2

                '''PART ADDED'''
                # mask loss
                self.loss_mask_ = 0
                self.loss_mask_A = 0
                self.loss_mask_B = 0
                if opt.loss_mode == 'mask':
                    self.loss_mask_A = criterion_mask(fake_B.mul(A_mask), white.mul(A_mask))
                    self.loss_mask_B = criterion_mask(fake_A.mul(B_mask), white.mul(B_mask))
                    if i%20==0:
                        #loss_mask_ = 0
                        #print(fake_B[0], real_A[0])
                        SaveFirstImage(real_A.mul(A_mask), 'mask.png')
                        SaveFirstImage(white.mul(A_mask), 'white.png')
                        SaveFirstImage(fake_A, 'fake.png')
                    self.loss_mask_ = (self.loss_mask_B + self.loss_mask_A)/2
                    
                # perceptual loss
                
                '''END'''
                # Cycle loss
                recov_A = self.G_BA(fake_B)
                recov_B = self.G_AB(fake_A)
                
                self.loss_cycle_A = criterion_cycle(recov_A, real_A)
                self.loss_cycle_B = criterion_cycle(recov_B, real_B)
                self.loss_cycle = (self.loss_cycle_A + self.loss_cycle_B)/2

                self.loss_perc_A = criterion_per(fake_B, real_A)
                self.loss_perc_B = criterion_per(fake_A, real_B)
                self.loss_perc = (self.loss_perc_A + self.loss_perc_B)/2

                recov_A = 0
                recov_B = 0
                #fake_A = 0
                #fake_B = 0

                # Total loss
                self.loss_G = self.loss_GAN + opt.lambda_cyc * self.loss_cycle + opt.lambda_id * self.loss_identity \
                        + opt.lambda_perc * self.loss_perc #+ 8* self.loss_line
                if opt.loss_mode == 'mask':
                    self.loss_G = self.loss_G + opt.lambda_mask * self.loss_mask_
                self.loss_G.backward()
                optimizer_G.step()

                # -----------------------
                #  Train Discriminator A
                # -----------------------

                optimizer_D_A.zero_grad()

                # Real loss
                loss_real = criterion_GAN(D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
                # Total loss
                self.loss_D_A = (loss_real + loss_fake) / 2

                self.loss_D_A.backward()
                optimizer_D_A.step()

                # -----------------------
                #  Train Discriminator B
                # -----------------------

                optimizer_D_B.zero_grad()

                # Real loss
                loss_real = criterion_GAN(D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
                # Total loss
                self.loss_D_B = (loss_real + loss_fake) / 2

                self.loss_D_B.backward()
                optimizer_D_B.step()

                self.loss_D = (self.loss_D_A + self.loss_D_B) / 2

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(self.dataloader) + i
                batches_left = opt.n_epochs * len(self.dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f,] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(self.dataloader),
                        self.loss_D.item(),
                        self.loss_G.item(),
                        self.loss_GAN.item(),
                        self.loss_cycle.item()*opt.lambda_cyc,
                        self.loss_identity.item()*opt.lambda_id,
                        #self.loss_mask_.item()*opt.lambda_mask,
                        time_left,
                    )
                )

                # If at sample interval save image
                if batches_done % opt.sample_interval == 0:
                    self.sample_images(batches_done, opt)
                    loss_item = self.get_current_losses()
                    for i in loss_item:
                        self.train_writer.add_scalar("%s"%(i), loss_item[i], batches_done*opt.batch_size)
                    
            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
                # Save model checkpoint
                torch.save(self.G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.outf, epoch))
                torch.save(self.G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.outf, epoch))
                torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.outf, epoch))
                torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.outf, epoch))


#run the model.
model = CYCLEGAN(opt)
if opt.run_mode=='train':
    model.train(opt)
elif opt.run_mode=='test':
    model.test(opt)
else:
    model.get_result(opt)
