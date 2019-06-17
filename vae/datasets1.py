from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from skimage import segmentation
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def erosion(img, kernalSize):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
    (kernalSize, kernalSize))
    ret,img = cv2.threshold(img,220,1,cv2.THRESH_BINARY)
    mask = cv2.erode(img, kernel, iterations=1)
    return mask

def dilation(img, kernalSize):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
    (kernalSize, kernalSize))
    ret,img = cv2.threshold(img,220,1,cv2.THRESH_BINARY)
    mask = cv2.dilate(img, kernel, iterations=1)
    return mask

# def erosion(img, radius):
#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.dilate(img, kernel, iterations=radius//2)
#     return mask

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def getImage(filename):
    import cv2
    src = cv2.imread(filename,cv2.IMREAD_COLOR)
    img = src.transpose(2,0,1).astype(np.float32)/255.0
    img = Variable(torch.from_numpy(img))
    return img, src

class DataDataset(data.Dataset):
    def __init__(self, root, crop_size, train=True):
        self.data = make_dataset(root)
        self.len = len(self.data)
        self.trans = get_transform(crop_size, train=train)

    def __getitem__(self, index):
        filepath = self.data[index%self.len]
        #print(filepath)
        #A,idx = getImage(filepath)
        A = Image.open(filepath).convert('L')
        A = self.trans(A)
        return A

    # for test
    # def __getitem__(self, index):
    #     filepath = self.data[index%self.len]
    #     A = Image.open(filepath).convert('L')
    #     # A = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    #     if A.size[0] >4000:
    #         A = A.resize((A.size[0]//2, A.size[1]//2), Image.BICUBIC)
    #         # A = cv2.resize(A, (A.shape[1]//2, A.shape[0]//2), interpolation=cv2.INTER_LINEAR)
    #     Asize = A.size
    #     A = self.trans(A)
    #     return A, filepath, Asize

    def __len__(self):
        return self.len


def get_transform(crop_size, train=True):
    transform_list = []
    if train:
        transform_list.append(transforms.RandomCrop(crop_size))
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=2**8)))
        
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, ),
                                            (0.5, ))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

# just modify the width and height to be multiple of 4
def __adjust(img, mult):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    # mult = 32
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __scale_width(img, target_width):
    ow, oh = img.size

    # the size needs to be a multiple of this number,
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 32
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
