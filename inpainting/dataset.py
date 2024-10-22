import os
import sys
import cv2
import PIL
import torch
import torch.utils.data as Data
import numpy as np

import torchvision
import torchvision.transforms as standard_transforms
from utils import *

from skimage import io, transform

from config import *

trans = standard_transforms.Compose([
            standard_transforms.CenterCrop(IMGSIZE),
            standard_transforms.ToTensor()])

class Manga109(Data.Dataset):
    def __init__(self, mode, root_dir, mask_dir,
     target_dir=None, line_dir=None, sct_dir=None,
     image_transform=None, target_transform=None,
     singletrans=None):

        self.matches = []
        self.mask_dir = mask_dir
        self.line_dir = line_dir
        self.target_dir = target_dir
        self.sct_dir = sct_dir
        self.transform = image_transform
        self.target_transform = target_transform
        self.singletrans = singletrans

        self.mode = mode
        self.root_dir = root_dir
        count = 0
        for _, _, filenames in os.walk(root_dir):
            for filename in filenames:
                self.matches.append(filename)
                count+=1
                if count == FILENUM:
                    break

            if count == FILENUM:
                break

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.matches[index])
        mask_name = os.path.join(self.mask_dir, FileName(self.matches[index], '.png'))
        screentone_name = os.path.join(self.sct_dir, FileName(self.matches[index], '.png'))

        self.mask = PIL.Image.open(mask_name).convert('L')
        self.image = PIL.Image.open(img_name).convert('RGB')

        if self.target_dir:
            target_name = os.path.join(self.target_dir, FileName(self.matches[index], '.png'))
            self.target_image = PIL.Image.open(target_name).convert('RGB')
            self.target_image = self.target_transform(self.target_image)
 
        line_name = os.path.join(self.line_dir, FileName(self.matches[index], '.png'))
        self.line = PIL.Image.open(line_name).convert('L')
        self.line = self.singletrans(self.line)

        if self.transform:
            self.shape = np.array(self.image).shape[0:2][::-1]
            self.image = self.transform(self.image)
            self.mask = self.singletrans(self.mask)
            if self.mode == 'valid':
                self.shape = np.array(self.image).shape[1:3][::-1]
            self.screentone = Image.fromarray(cv2.resize(GetImg(screentone_name, PATH_GRAY),
                                             self.shape))
            if self.mode != 'valid':
                self.screentone = trans(self.screentone)

            else:
                tran = standard_transforms.ToTensor()
                self.screentone = tran(self.screentone)
            print(np.array(self.image).transpose(1,2,0).shape, np.array(self.mask).shape,
                  np.array(self.screentone).shape, self.shape)

            self.train_data = cv2.merge((np.array(self.image).transpose(1,2,0),
                                         np.array(self.line).transpose(1,2,0), 
                                         np.array(self.mask).transpose(1,2,0),
                                         np.array(self.screentone).transpose(1,2,0)
                                         ))
            if CUDA:
                self.train_data = torch.FloatTensor(self.train_data.transpose((2, 0, 1))).cuda()
            else:
                self.train_data = torch.FloatTensor(self.train_data.transpose((2, 0, 1)))

        if self.mode == 'train':
            return self.train_data, self.target_image, self.mask, self.line
        elif self.mode == 'test':
            return self.train_data, self.target_image, self.mask, self.line
        else:
            return self.train_data, self.mask, self.line

class InPaintLoader:
    def __init__(self, mode):
        self.mode = mode

        # VGG mean #
        mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

        self.transform = standard_transforms.Compose([
            standard_transforms.CenterCrop(IMGSIZE),
            FlipChannels(),
            standard_transforms.ToTensor(),
            #standard_transforms.Lambda(lambda x: x.mul_(255)),
            #standard_transforms.Normalize(*mean_std)
        ])

        self.singlelayer_transform = standard_transforms.Compose([
            standard_transforms.CenterCrop(IMGSIZE),
            Binarize(),
            standard_transforms.ToTensor(),
            #standard_transforms.Lambda(lambda x: x.mul_(255)),
        ])

        self.singlelayer_test =  standard_transforms.Compose([
            SingleResize(),
            Binarize(),
            standard_transforms.ToTensor(),
            #standard_transforms.Lambda(lambda x: x.mul_(255)),
        ])

        self.target_transform = standard_transforms.Compose([
            standard_transforms.CenterCrop(IMGSIZE),
            FlipChannels(),
            standard_transforms.ToTensor()
        ])

        self.target_test =  standard_transforms.Compose([
            SingleResize(),
            FlipChannels(),
            standard_transforms.ToTensor()
        ])

        self.test_transform = standard_transforms.Compose([
            SingleResize(),
            FlipChannels(),
            standard_transforms.ToTensor(),
            standard_transforms.Lambda(lambda x: x.mul_(255)),
            standard_transforms.Normalize(*mean_std)
        ])

        self.restore = standard_transforms.Compose([
            DeNormalize(*mean_std),
            standard_transforms.Lambda(lambda x: x.div_(255)),
            standard_transforms.ToPILImage(),
            FlipChannels()
        ])

        if self.mode == 'train':
            train_dataset = Manga109(self.mode, INPUT, MASKDIR, target_dir=OUTPUT, sct_dir=SCTDIR,
                line_dir=ROOTDIR, image_transform=self.transform, target_transform=self.target_transform,
                singletrans=self.singlelayer_transform)

            self.train_loader = Data.DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )
            self.length = len(self.train_loader)

        elif self.mode == 'test':
            if TEST_BATCH_SIZE == 1:
                test_dataset = Manga109(self.mode, T_INPUT, T_MASKDIR,
                    target_dir=T_OUTPUT, line_dir=T_ROOTDIR, sct_dir=T_SCTDIR,
                    image_transform=self.test_transform,
                    singletrans=self.singlelayer_test,
                    target_transform=self.target_test)
            else:
                test_dataset = Manga109(self.mode, T_INPUT, T_MASKDIR, 
                    target_dir=T_OUTPUT, line_dir=T_ROOTDIR, sct_dir=T_SCTDIR,
                    image_transform=self.transform,
                    singletrans=self.singlelayer_transform,
                    target_transform=self.target_transform)
    
            self.test_loader = Data.DataLoader(
                test_dataset,
                batch_size=TEST_BATCH_SIZE,
                shuffle=False,
                num_workers=0
            )
            self.length = len(self.test_loader)

        elif self.mode == 'valid':
            valid_dataset = Manga109(self.mode, V_INPUT, V_MASKDIR,
                    line_dir=V_ROOTDIR, sct_dir=V_SCTDIR,
                    image_transform=self.test_transform,
                    singletrans=self.singlelayer_test,
                    target_transform=self.target_test)

            self.valid_loader = Data.DataLoader(
                valid_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            self.length = len(self.valid_loader)
