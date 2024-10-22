import os
import time
import torch
from random import randint
import cv2
from PIL import Image

import numpy as np

IMG = 0
PATH_COLOR = -1
PATH_GRAY = 1
start_time = 0



def getResultPic(line, image):
    line = cv2.resize(line, image[0:2][::-1], interpolation=cv2.INTER_NEAREST)
    image[line < 0.1] = 0
    return image

def realSize(size):
    return (size[0] - size[0]%16, size[1] - size[1]%16)

class FlipChannels(object):
    def __call__(self, img):
        if len(img.size) != 3:
            return img
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))

class Binarize(object):
    def __call__(self, img, screentone=None):
        if screentone:
            return img
        img = np.array(img)
        _, img = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
        #print(img)
        return Image.fromarray((img).astype(np.uint8))

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
        
class SingleResize(object):
    def __call__(self, img):
        img = np.array(img)
        newshape = realSize(img.shape[0:2][::-1])
        print(newshape)
        img = cv2.resize(img, newshape)
        return Image.fromarray(img.astype(np.uint8))

def SaveFirstImage(tensor, path='./test.png'):
    img = tensor[0].cpu().detach().numpy().transpose(1,2,0)
    cv2.imwrite(path, np.uint8(img*255))
    return

def Hex2RGB(strInput):
    return tuple(int(strInput[i:i + 2], 16) for i in (0, 2, 4))

def loadImages(folder):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            matches.append(os.path.join(root, filename))
    return matches

def FileName(path, extention):
    return os.path.splitext(os.path.basename(path))[0] + extention

def GetImg(imgPath, mode):
    if mode == IMG:
        img = imgPath
    if mode == PATH_COLOR:
        img = cv2.imread(imgPath)
    if mode == PATH_GRAY:
        img = cv2.imread(imgPath, 0)
    return img

def ShowAndWait(img, name='testing'):

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clock(notice):
    if notice == "start":
        print('start')
    else:
        global start_time
        print(notice, 1000*(time.time() - start_time), "ms")
    
    start_time = time.time()


def Erosion(imgPath, kernalSize, mode):
    img = GetImg(imgPath, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernalSize, kernalSize))
    ret,erosion = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    erosion = cv2.erode(img, kernel, iterations=1)
    return erosion

def Dilation(imgPath, kernalSize, mode):
    img = GetImg(imgPath, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernalSize, kernalSize))
    ret,dilation = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    dilation = cv2.dilate(img, kernel, iterations=1)
    return dilation

#FloodFill function, define number of color values.
def FloodFill(imgPath, numOfLevel, mode):

    colorlevels = []
    COLOR = lambda: randint(0,255)
    for _ in range(numOfLevel):
        colorlevels.append((COLOR(), COLOR(), COLOR()))

    img = GetImg(imgPath, mode)
    height, width = img.shape[0:2]
    upper = (10, 10, 10)
    lower = (10, 10, 10)
    mask = np.zeros((height + 2, width + 2), np.uint8)
    count = 0

    for i in range(0, height):
        for j in range(0, width):
            # in cv2.floodFill(), the coordinate of seed point
            # is (x, y), follows Cartesian system.
            if np.array_equal(img[i][j], [255, 255, 255]):
                color = colorlevels[randint(0, numOfLevel - 1)]
                count += 1
                cv2.floodFill(img, mask, (j, i), color, lower,
                              upper)
                #ShowAndWait(img)
    return img
