import i2v
from PIL import Image

import torch
import os
import time
import numpy as np
from datasets import DataDataset

dataset = '../../../data/manga'
books = os.listdir(os.path.join(dataset,'simline'))
books.sort()
name = 'feature5'
start = time.time()

#dataset = DataDataset(dataset, base=0, name=name)
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
#                                        shuffle=False, num_workers=1)

illust2vec = i2v.make_i2v_with_caffe("illust2vec_ver200.caffemodel")

test = Image.open('../test/001.jpg')
result_real = illust2vec.extract_feature([test])
print(result_real)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.bmp', '.BMP',
    '.npy',
]

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

images = make_dataset(os.path.join(dataset,'simline'))
result = []
print(len(images))
for i,img in enumerate(images):
    data = Image.open(img)
    if i%100==0:
        print(i)
    result_real = illust2vec.extract_feature([data])
    #print(result_real)
    result.append(result_real[0])
    #print(result)

np.save('4096.npy', np.asarray(result))
#img = Image.open("images/miku.jpg")

# extract a 4,096-dimensional feature vector

#print("shape: {}, dtype: {}".format(result_real.shape, result_real.dtype))
#print(result_real)

# i2v also supports a 4,096-bit binary feature vector
#result_binary = illust2vec.extract_binary_feature([img])
#print("shape: {}, dtype: {}".format(result_binary.shape, result_binary.dtype))
#print(result_binary)