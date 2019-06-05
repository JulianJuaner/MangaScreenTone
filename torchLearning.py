#Sample CNN.
import os
import sys
import cv2
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import lr_scheduler
import torchvision
from tensorboardX import SummaryWriter
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from skimage import io, transform


#visualization of process.
from tqdm import tqdm

writer = SummaryWriter()

EPOCH = 10
LR = 1e-4
BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
IMGSIZE = (256, 256)
FILENUM = 400
CUDA = True

start_time = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if CUDA == False:
    FILENUM = 200
    IMGSIZE = (128, 128)

class ImageVOC(Data.Dataset):
  def __init__(self, root_dir, transform = None, mode = 'train'):
        self.root_dir = root_dir
        self.matches = []
        self.transform = transform
        self.mode = mode
        count = 0
        for root, dirnames, filenames in os.walk(root_dir):
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
        self.image = cv2.imread(img_name)
        self.gray_image = cv2.imread(img_name, 0)
        self.imgshape = torch.Tensor(np.asarray(self.image.shape))

        if CUDA:
            self.imgshape = self.imgshape.cuda()

        if self.transform:
            self.image, self.gray_image = self.transform(self.image, self.gray_image)

        if self.mode == 'train':
            return self.image, self.gray_image

        elif self.mode == 'test':
            return self.image, self.gray_image, self.imgshape

def normalizeImg(img):

    return cv2.normalize(np.absolute(img), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

class PreProcess(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img, gray_image):
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = (cv2.resize(img, (new_h, new_w))).astype(float)/255
        gray_image = (cv2.resize(gray_image, (new_h, new_w))).astype(float)/255
        if CUDA:
            return torch.FloatTensor(img.transpose((2, 0, 1))).cuda(), torch.FloatTensor(gray_image).cuda().unsqueeze(0)
        else:
            return torch.FloatTensor(img.transpose((2, 0, 1))), torch.FloatTensor(gray_image).unsqueeze(0)
            
class FirstTryEncoder(nn.Module):
    def __init__(self):
        super(FirstTryEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64, momentum=0.5),
            #nn.MaxPool2d(kernel_size=2), 
            #nn.Conv2d(64,64,3,1,1),
            #nn.ReLU(),
            #nn.Conv2d(64,64,3,1,1),
            #nn.ReLU(),
            #nn.Dropout(0.3),
            #nn.BatchNorm2d(64, momentum=0.5),
            #nn.MaxPool2d(kernel_size=2), 
            nn.Conv2d(64,32,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.Conv2d(32,16,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(16, momentum=0.5),
            nn.Conv2d(16,4,3,1,1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm2d(4, momentum=0.5),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(4, 1, 3, 1, 1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(1,4,3,1,1),
            nn.ReLU(),
            nn.Conv2d(4,16,3,1,1),
            #nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm2d(16, momentum=0.5),
            nn.Conv2d(16,64,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64, momentum=0.5),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.Conv2d(64,16,3,1,1),
            nn.ReLU(),
            nn.Dropout(0.3),
            #nn.BatchNorm2d(16, momentum=0.5),
            nn.Conv2d(16,3,3,1,1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def clock(notice):
    global start_time
    print(notice, ", ", 1000*(time.time() - start_time), "ms")
    start_time = time.time()

def train(dataloader, testloader):
    global start_time
    best_acc = 0.0

    encoder = FirstTryEncoder()
    
    if CUDA:
        encoder.cuda()

    enoptimizer = torch.optim.Adam(encoder.parameters(), lr = LR)
    en_scheduler = lr_scheduler.StepLR(enoptimizer, step_size=10, gamma=0.5)

    loss1 = nn.MSELoss()
    iteration = 0

    for epoch in range(EPOCH):
        en_scheduler.step()

        running_loss = 0.0
        clock('EPCOH: {}/{}'.format(epoch + 1, EPOCH))

        for step, (imgs, grayimgs) in enumerate(tqdm(dataloader)):
            #print(step)
            output = encoder(imgs)
            loss = loss1(output, imgs)
            output = 0
            print(loss.item())
            writer.add_scalar('data/loss', loss.item(),
             iteration)
            iteration += 1

            enoptimizer.zero_grad()

            loss.backward()                
            enoptimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(dataloader)
        print('\ntrain epoch Loss: {:.4f} '.format(epoch_loss))

        running_loss = 0.0
        for test_step, (test_imgs, test_grayimgs, origin_size) in enumerate(testloader):

            test_output = encoder(test_imgs)
            loss = loss1(test_output, test_imgs)
            running_loss += loss.item() * test_imgs.size(0)
            loss = 0
            print("test_loss: {:.4f}".format(running_loss / len(testloader)))

            resultI = test_output.cpu().detach().numpy().transpose(0,2,3,1)
            origin_size = origin_size.cpu().detach().numpy().astype(int)

            for step, (img) in enumerate(resultI):
                Iimg = cv2.resize(255*(img),
                 tuple(origin_size[step][:2][::-1]), interpolation=cv2.INTER_LINEAR).astype(int)

                emm = cv2.resize(255*test_imgs.cpu().detach().numpy().transpose(0,2,3,1)[step], 
                                tuple(origin_size[step][:2][::-1]), interpolation=cv2.INTER_LINEAR).astype(int)

                cv2.imwrite('./out/{:02d}resultI.jpg'.format(step), Iimg)
                cv2.imwrite('./out/{:02d}resultG.jpg'.format(step), emm)

    writer.export_scalars_to_json("./test.json")
    writer.close()
    torch.save(encoder.state_dict(), './model/encoder.pkl')


if __name__ == "__main__":
    start_time = time.time()

    image_dataset = ImageVOC('./img/JPEGImages/', 
        transform = PreProcess(IMGSIZE))
    test_image_dataset = ImageVOC('./img/testIMG/', 
        transform = PreProcess(IMGSIZE), mode = 'test')

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(test_image_dataset, batch_size=TEST_BATCH_SIZE,
                        shuffle=False, num_workers=0)

    clock("dataloading")
    train(dataloader, testloader)

