import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np
import cv2
IMG = 0
PATH_COLOR = -1
PATH_GRAY = 1
start_time = 0

from torchvision.utils import save_image

def GetImg(imgPath, mode):
    if mode == IMG:
        img = imgPath
    if mode == PATH_COLOR:
        img = cv2.imread(imgPath)
    if mode == PATH_GRAY:
        img = cv2.imread(imgPath, 0)
    return img

def Threshold(img, mode=IMG, threshold=220):
    img = GetImg(img, mode)
    ret,img = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    return img

def Erosion(imgPath, kernalSize=3, mode=IMG):
    img = GetImg(imgPath, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernalSize, kernalSize))
    ret,erosion = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
    erosion = cv2.erode(erosion, kernel, iterations=1)
    _, erosion = cv2.threshold(erosion,127,255,cv2.THRESH_BINARY)
    return erosion
    
def Dilation(imgPath, kernalSize=3, mode=IMG):
    img = GetImg(imgPath, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernalSize, kernalSize))
    ret,dilation = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    dilation = cv2.dilate(img, kernel, iterations=2)
    return dilation

def SaveFirstImage(tensor, path='./test.png'):
    img = tensor[0].cpu().detach().numpy().transpose(1,2,0)
    cv2.imwrite(path, np.uint8(img*255))
    return

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
