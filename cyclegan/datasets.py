import glob
import os
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data

from utils import *
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def to_grayscale(image):
    gsc_image = Image.new("L", image.size)
    gsc_image.paste(image)
    return gsc_image


class ImageDataset(Dataset):
    def __init__(self, rootA, rootB, transforms_=None, datasize = None,
     unaligned=False, mode="train", channel = 3, mask = False, opt = None):
        print(rootA, rootB)
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.datasize = datasize
        self.channel = channel
        self.mode = mode
        self.mask = mask
        self.files_A = sorted(glob.glob(os.path.join(rootA, "%s/lineA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(rootB, "%s/B" % mode) + "/*.*"))
        self.opt = opt

        if self.mask:
            mask_A = '../../data/manga' #+ self.mode + '/mask_A/'
            mask_B = '../../data/manga' #+ self.mode + '/mask_B/'
            self.mask_A = sorted(glob.glob(os.path.join(mask_A, "%s/A_mask" % mode) + "/*.*"))
            self.mask_B = sorted(glob.glob(os.path.join(mask_A, "%s/B_mask" % mode) + "/*.*"))

    def __getitem__(self, index):
        imgs = []
		#masks = []
        if self.datasize:
            self.size=random.randint(self.opt.num_img-4, self.opt.num_img+2)
            for i in range(self.size):
                random_num = random.randint(0, (len(self.files_A) - 1))#imgs.append(cv2.imread(self.files_A[index % self.opt.num_img*self.datasize], 0))
                imgs.append(Image.open(self.files_A[random_num]))
				#masks.append(Image.open(self.mask_A[random_num]))
        else:
            image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            if self.datasize:
                image_B = Image.open(self.files_B[random.randint(0, (self.datasize - 1))])
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            if self.datasize:
                image_B = Image.open(self.files_B[index % self.datasize])
            image_B = Image.open(self.files_B[index % len(self.files_B)])
        
        # Convert grayscale images to rgb
        if self.channel == 3:
            if image_A.mode != "RGB":
                image_A = to_rgb(image_A)
            if image_B.mode != "RGB":
                image_B = to_rgb(image_B)
        else:
            for i in range(self.size):
                imgs[i] = to_grayscale(imgs[i])
            image_B = to_grayscale(image_B)

        A_transform = []
        items_A = []
        masks_A = []
        item_A = np.zeros((256,256,1)).astype('uint8')
        mask_A = np.zeros((256,256,1)).astype('uint8')
        for i in range(self.size):
            params = get_params(self.opt, imgs[i].size)
            A_transform.append(get_transform(self.opt, grayscale=(self.channel == 1), mode='A', normal=False))
            items_A.append(~((255*np.asarray(A_transform[i](imgs[i]))).astype('uint8').transpose(1,2,0)))
            masks_A.append(Dilation(items_A[i]))
            mask_A = cv2.add(mask_A, masks_A[i])
            #print(items_A[i].shape, item_A.shape)
            item_A = cv2.add(item_A, items_A[i])
        B_transform = get_transform(self.opt, params, grayscale=(self.channel == 1), normal=False)
        #fulltrans = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        item_A = get_transform(self.opt, grayscale=(self.channel == 1), mode='C')(Image.fromarray(~item_A))
        mask_A = get_transform(self.opt, grayscale=(self.channel == 1), mode='C')(Image.fromarray(~mask_A))
        item_B = B_transform(image_B)

        if self.mask:
            #if self.datasize:
                #masks = []
                #for i in range(self.opt.num_img):
                    #masks.append(Image.open(self.mask_A[index % self.opt.num_img*self.datasize]))
                #print(self.mask_A)
                #mask_A = Image.open(self.mask_A[index % self.datasize])
            #else:
                #mask_A = Image.open(self.mask_A[index % len(self.mask_A)])
            if self.unaligned:
                if self.datasize:
                    mask_B = Image.open(self.mask_B[random.randint(0, (self.datasize - 1))])
                mask_B = Image.open(self.mask_B[random.randint(0, len(self.mask_B) - 1)])
            else:
                if self.datasize:
                    mask_B = Image.open(self.mask_B[index % self.datasize])
                mask_B = Image.open(self.mask_B[index % len(self.mask_B)])

            #mask_trans_A = get_transform(self.opt, params, grayscale=(self.channel == 1), normal=False)
            mask_trans_B = get_transform(self.opt, params, grayscale=(self.channel == 1), normal=False)
            #mask_A = to_grayscale(mask_A)
            mask_B = to_grayscale(mask_B)
            
            maskA = mask_A
            maskB = mask_trans_B(mask_B)
            
            return {"A": item_A, "B": item_B, "maskA":maskA, "maskB":maskB}

        #print(item_A.size(), item_B.size())
        return {"A": item_A, "B": item_B}

    def __len__(self):
        if self.datasize:
            return self.datasize
        return max(len(self.files_A), len(self.files_B))

#Functions for gen transforms.
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.bmp', '.BMP',
    '.npz',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    already=0
    i=0
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                #img = GetImg(os.path.join(root, fname), PATH_GRAY)
                #if img.shape != (1170, 827):
                    #print(img.shape)
                    #print(os.path.join(root, fname))
                #else:
                if i >= already:
                    path = os.path.join(root, fname)
                    images.append(path)
                i+=1
                #print(path)
    images.sort()
    return images
    
#Test dataset...(whole image.)
class TestDataset(Dataset):
    def __init__(self, file_list, opt):
        self.opt = opt
        self.file_list = make_dataset(file_list)
        self.transform = test_transform()

    def __getitem__(self, index):

        #image = Image.fromarray(Threshold(GetImg(self.file_list[index], PATH_GRAY)))
        image = Image.open(self.file_list[index])
        image = self.transform(image)
        return {"Name": self.file_list[index], "Image": image}


    def __len__(self):
        return len(self.file_list)

def test_transform():
    return transforms.Compose([
            transforms.Grayscale(1),
            transforms.Lambda(lambda img: __make_power_2(img, base=4, method=Image.BICUBIC)),
            transforms.ToTensor()
        ])

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=True, method=Image.BICUBIC, convert=True, normal=False, mode='B'):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        if mode == 'B':
            osize = [opt.load_size, opt.load_size]
            #transform_list.append(transforms.Resize(osize, method))
    elif ('scale_width' in opt.preprocess):
        if mode == 'B':
            i=1
            #transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    if mode =='A':
        transform_list.append(transforms.Pad(100, fill=255))#, padding_mode='constant'))
        #transform_list.append(transforms.RandomRotation(180))
        #transform_list.append(transforms.CenterCrop(300))
        #transform_list.append(transforms.Pad(50, fill=255, padding_mode='constant'))
    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            i=0
            transform_list.append(transforms.CenterCrop(opt.crop_size))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if normal:
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
    

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
