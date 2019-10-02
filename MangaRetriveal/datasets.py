from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import random
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from skimage import segmentation
import skimage
import cv2
import scipy.sparse as sparse

MAX_POOL = 2
#OFFLINE = True
GAUSSIAN = False

def low_reso(img):
	return skimage.measure.block_reduce(img, (2,2), np.max)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.bmp', '.BMP',
    '.npy', '.npz', '.svg','.pt'
]
TRANS = transforms.Compose([transforms.ToTensor()])
tran = TRANS

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    print(dir)
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    images.sort()
    return images

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
crop = transforms.RandomCrop((256, 256))

def cropresize():
    randsize = random.randint(200, 400)
    randcrop = transforms.RandomCrop((randsize, randsize))
    resize = transforms.Resize((256, 256), interpolation=2)
    return randcrop, resize


def gaussBlur(img, kern_size=(21,21)):
    return cv2.GaussianBlur(img, kern_size, 0)

def getImage(filename, scale=1, mode=0):
    src = cv2.imread(filename,1)
    if mode == 2:
        C, R = cropresize()
        src = Image.fromarray(src)
        srcC = C(src)
        srcR = R(srcC)
        srcR = np.array(srcR)
        srcC = np.array(srcC)
        imgR = srcR.transpose(2,0,1).astype(np.float32)
        imgR = Variable(torch.from_numpy(imgR), requires_grad=False)
        imgC = srcC.transpose(2,0,1).astype(np.float32)
        imgC = Variable(torch.from_numpy(imgC), requires_grad=False)
        return imgC, imgR

    if mode==1:
        src = cv2.resize(src, (256, 256))
    if GAUSSIAN and mode==1:
        src = gaussBlur(src)
        if mode==1:
            cv2.imwrite('smooth.jpg', src)
    img = src.transpose(2,0,1).astype(np.float32)
    img = Variable(torch.from_numpy(img), requires_grad=False)

    return img
def getFeature(filename, normal=False, channel=32):
    feature = torch.load(filename)#[channel:, :, :]
    #feature = torch.FloatTensor(feature)
    return Variable(feature,requires_grad=False)

def Erosion(img, kernalSize=3):
    #img = GetImg(imgPath, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernalSize, kernalSize))
    ret,erosion = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    erosion = cv2.erode(img, kernel, iterations=1)
    ret,erosion = cv2.threshold(erosion,127,255,cv2.THRESH_BINARY)
    return erosion


class DataDataset(data.Dataset):
    def __init__(self, root, base=32, chnls=512, scale=1.0, mode='img', name='featureHalfNC', file_list=None, basic=0, length=-1):
        #print(mode)
        
            #print('here is feature')
        self.file_list = file_list
        self.feat = make_dataset(os.path.join(root,name))
        if file_list:
            print('feature length:', len(self.file_list))
        if 'compress' in mode:
            self.kernset = make_dataset(os.path.join(root, 'trainA3'))
        if 'focus' in name:
            self.data = make_dataset(os.path.join(root,'focus'))
        else:
            self.data = make_dataset(os.path.join(root,'simline'))
        self.len = len(self.data)
        print('length:', self.len, os.path.join(root,name))
        self.basic = basic
        self.mode = mode
        self.base = base
        self.chnls = chnls
        self.scale = scale

    def __getitem__(self, index):
        
        if self.mode=='img':
            if self.file_list:
                filepath = self.feat[self.file_list[index]%self.len]
            else:
                #print(index, self.len)
                filepath = self.feat[index%self.len]
            if self.basic==1:
                print(filepath)
            return getImage(filepath, self.scale, self.basic)
        elif self.mode=='none':
            return [0]
        elif 'feature_compress' in self.mode:
            rand1 = random.randint(0, self.len-1)
            rand2 = random.randint(0, 12000)
            #print(rand1, rand2)
            feat1 = self.feat[rand1]
            feat2 = self.kernset[rand2]
            #print(feat1, feat2)
            #--------------------------------
            if "resize" in self.mode:
                img1 = getImage(feat1,1)
                img2, img3 = getImage(feat1, mode=2)
                return [img1, img2, img3]
            #--------------------------------
            return [getImage(feat1,1,0), getImage(feat2,1,0)]
            
        else:
            #filepath = self.data[index%self.len]
            filepath2 = self.feat[index%self.len]
            return [getFeature(filepath2)]

    def __len__(self):
        if self.file_list:
            #print(len(self.file_list))
            return len(self.file_list)
        return self.len
