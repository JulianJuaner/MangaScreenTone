import numpy as np
import os
#import cv2
#import torch
d = 4096                       # dimension
nb = 20936                      # database size
nq = 1                         # nb of queries

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

filename = make_dataset('../../../../data/manga/simline')
print(len(filename))
db = np.load('4096.npy')
query = np.load('query.npy')
query[query<0] = 0
print(query)
for i in range(len(db)):
    db[i][query[0]==0] = 0
print('ok?')
print(query.shape)

import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index

print(index.is_trained)
index.add(db)                  # add vectors to the index
print(index.ntotal)

k=10
#D, I = index.search(db[:5], k) # sanity check
#print(I)
#print(D)
D, I = index.search(query, k)     # actual search
print(I[:1])                   # neighbors of the 5 first queries
print(I[-1:])                  # neighbors of the 5 last queries

for item in I[0]:
    print(filename[item])
'''    img = cv2.imread(filename[item])
    cv2.imwrite('%d.jpg'.format(item), img)'''


