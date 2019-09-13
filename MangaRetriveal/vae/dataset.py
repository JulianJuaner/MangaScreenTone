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
from sklearn.preprocessing import minmax_scale
import cv2
import scipy.sparse as sparse

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.bmp', '.BMP',
    '.npz', '.npy',
]
#TRANS = transforms.Compose([transforms.Normalize(0.5)])
#tran = TRANS

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

def getImage(filename, scale):
    src = cv2.imread(filename,cv2.IMREAD_COLOR)
    src = cv2.resize(src, None, fx=scale,fy=scale, interpolation=cv2.INTER_LINEAR)
    
    img = src.transpose(2,0,1).astype(np.float32)
    img = Variable(torch.from_numpy(img),requires_grad=False)
    return img

def getFeature(filename, normal=False):
    #feature = (sparse.load_npz(filename).toarray())
    #print(feature)
    #print(np.max(feature), np.mean(feature))
    #feature = np.array(feature)
    feature = np.load(filename)#.reshape(1, -1)
    #feature = minmax_scale(feature, axis=1)
    #print(np.max(feature), np.mean(feature))
    #feature = feature.reshape((512, 73, 51))
    #print(np.max(feature), np.mean(feature))
    feature = torch.FloatTensor(feature)
    #feature = torch.nn.functional.normalize(feature)
    #sprint(feature)
    #if normal == True:
        #feature = tran(feature)
    feature = Variable(feature,requires_grad=False)
    return feature

class DataDataset(data.Dataset):
    def __init__(self, root, base=32, chnls=512, scale=1.0, mode='img', name='simline', length=3000):
        #print(mode)
        if mode !='img':
            #print('here is feature')
            self.feat = make_dataset(os.path.join(root,name))#[:length]
            print('feature length:', len(self.feat))
        if 'compress' in mode:
            self.kernset = make_dataset(os.path.join(root, 'trainA3'))
        self.data = make_dataset(os.path.join(root,'simline'))#[:length]
        self.len = len(self.data)
        print('length:', len(self.data))
        self.mode = mode
        self.base = base
        
        if length==3000:
            self.length = length
        else:
            self.length = self.len

        self.chnls = chnls
        self.scale = scale

    def __getitem__(self, index):
        
        if self.mode=='img':
            filepath = self.data[index%self.len]
            print(filepath)
            return getImage(filepath, self.scale)
        elif self.mode =='feature_compress':
            rand1 = random.randint(0, self.len-1)
            rand2 = random.randint(0, 12000)
            rand3 = random.randint(0, self.len-1)
            #print(rand1, rand2)
            feat1 = self.feat[rand1]
            feat2 = self.kernset[rand2]
            feat3 = self.feat[rand3]
            #print(feat1, feat2)
            return [getImage(feat1,1), getImage(feat2,1)]
            #pass
        else:
            filepath = self.data[index%self.len]
            filepath2 = self.feat[index%self.len]
            return [getFeature(filepath2)]

    def __len__(self):
        return self.length
