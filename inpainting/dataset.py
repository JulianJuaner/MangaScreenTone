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

class Manga109(Data.Dataset):
    def __init__(self, mode, root_dir, mask_dir, transform=None, target_transform=None):
        self.matches = []
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
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
        mask_name = os.path.join(self.mask_dir,os.path.basename(img_name))
        self.mask = PIL.Image.open(mask_name).convert('L')
        self.image = PIL.Image.open(img_name).convert('RGB')
        self.imgshape = torch.Tensor(np.asarray(self.image.shape))

        if CUDA:
            self.imgshape = self.imgshape.cuda()

        if self.transform:
            self.image = self.transform(self.image)
            self.mask = self.transform(self.mask)
            self.target_image = self.target_transform(self.image)

        return self.image, self.target_image, self.mask

class InPaintLoader:
    def __init__(self, mode):
        self.mode = mode

        # VGG mean #
        mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

        self.transform = standard_transforms.Compose([
            standard_transforms.CenterCrop(IMGSIZE),
            FlipChannels(),
            standard_transforms.ToTensor(),
            standard_transforms.Lambda(lambda x: x.mul_(255)),
            standard_transforms.Normalize(*mean_std)
        ])

        self.target_transform = standard_transforms.Compose([
            standard_transforms.CenterCrop(IMGSIZE),
            FlipChannels(),
            standard_transforms.ToTensor()
        ])

        self.test_transform = standard_transforms.Compose([
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
            train_dataset = Manga109(self.mode, './dataset/train/', self.transform, self.target_transform)
            self.train_loader = Data.DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )

        elif self.mode == 'test':
            if TEST_BATCH_SIZE == 1:
                test_dataset = Manga109(self.mode, './data/input/', self.test_transform)
            else:
                test_dataset = Manga109(self.mode, './data/valid/', self.transform)

            self.test_loader = Data.DataLoader(
                test_dataset,
                batch_size=TEST_BATCH_SIZE,
                shuffle=False,
                num_workers=0
            )
