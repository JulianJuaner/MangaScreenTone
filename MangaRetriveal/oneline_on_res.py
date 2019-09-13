from __future__ import print_function
import os
import cv2
import time
import argparse

import torch
import numpy as np

from datasets import DataDataset, low_reso, Erosion
from torch import nn
from rpn import Similarity
from models.networks import Illust2vecNet, myVGG
from vae.encoder import FeatureEncoder
from sklearn.decomposition import PCA
from tqdm import tqdm
from offline import two_addition
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#python online.py --batchSize 384 --gpu 1 --outf image/out --dataset image/manga --testimgs image/test --scales 0.7,1,1.5

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--outf', type=str, default='image/out',  help='output folder')
parser.add_argument('--dataset', type=str, default='../../../data/manga',  help='dataset mode')
parser.add_argument('--testimgs', type=str, default='image/test',  help='test images')
parser.add_argument('--scales', type=str, default='0.5,1,2',  help='scales of test images')
parser.add_argument('--start', type=int, default=300, help='start epoch of encoder')
parser.add_argument('--modelf', type=str, default='two_addition',  help='model folder')
simline = False

if simline:
    folder1 = 'simline'
    folder2 = 'low_reso2_sim'
    folder3 = 'simline'
else:
    folder1 = 'image'
    folder2 = 'low_reso2'
    folder3 = 'image'

folders = [folder1, folder2, folder3]

args = parser.parse_args()
#32_4_layer_p2p
device = torch.device("cuda")
torch.cuda.set_device(args.gpu)
#
chnls = 512
#pca = PCA(n_components=chnls)

base = 2**4

h,w = 1170//base,827//base
is_model = False

mb_size = args.batchSize
scales=[float(s)*0.5 for s in args.scales.split(',')]
illus2vec = Illust2vecNet('models/illust2vec_tag_ver200_2.pth').eval().cuda()
if is_model:
	featureEnc = FeatureEncoder().eval().cuda()
	featureEnc.load_state_dict(torch.load('vae/checkpoints/%s/model_%d.pth' % (args.modelf, args.start-1)))
################
#illus2vec=nn.DataParallel(illus2vec)
################

scale = 1
origindataset = DataDataset(args.dataset, base=base, chnls=chnls, mode='img', name=folders[0])
dataloader = torch.utils.data.DataLoader(origindataset, batch_size=args.batchSize,
                                         shuffle=False, num_workers=1)
print('Images in dataset: %s'%len(origindataset))
tdataset = DataDataset(args.testimgs, scale=scale, name='image', basic=1)
tdataloader = torch.utils.data.DataLoader(tdataset, batch_size=1,
                                         shuffle=False, num_workers=1)
feature = torch.FloatTensor(np.random.rand(mb_size, chnls, 73, 51)).cuda()
image1 = torch.FloatTensor(np.zeros((mb_size, 1, int(1170), int(827)))).cuda()
image2 = torch.FloatTensor(np.zeros((mb_size, 1, int(1170)//2, int(827/2)))).cuda()
image3 = torch.FloatTensor(np.zeros((mb_size, 1, int(1170/4), int(827/4)))).cuda()
images = [image3, image2, image1]

print("fake img generated")
nets = []
rois_s = []
scores_s = []
ret_s = [20, 300, 20]
scales = [1, 0.5, 1]
levels = [255-164.76139251,  255-167.47864617,  255-181.13838569]
for j, tdata in enumerate(tdataloader):
    file_list = [[],[],[]]
    #first level.
    for I in range(1):
        #print(tdata.shape)
        test_image = tdata[0].numpy().transpose(1,2,0)
        imagemask = Erosion(test_image, 30)
        for count1 in range(256):
            for count2 in range(256):
                #print(imagemask[count1][count2])
                if 255 in imagemask[count1][count2]:
                    for level in range(3):
                        imagemask[count1][count2][level] = levels[level]
        test_image = test_image #- imagemask

        #Attemption for cropping the image with a bounding rectangle.
        '''rows = np.amin(test_image, axis=0, keepdims=False)
        cols = np.amin(test_image, axis=1, keepdims=False)
        print(rows.shape, cols.shape)
        rect = [0,0,0,0]
        trigger = [False, False, False, False]
        for counter in range(256):
            if trigger[0] == False and rows[counter][0]!=255:
                rect[0] = counter
                trigger[0] = True
            if trigger[1] == False and rows[255-counter][0]!=255:
                rect[1] = 256-counter
                trigger[1] = True
            if trigger[2] == False and cols[counter][0]!=255:
                rect[2] = counter
                trigger[2] = True
            if trigger[3] == False and cols[255-counter][0]!=255:
                rect[3] = 256-counter
                trigger[3] = True
            if trigger == [True, True, True, True]:
                break
        print('get bounded crop value:', rect)
        test_image = test_image[rect[2]:rect[3], rect[0]:rect[1]]'''
        #test_image = np.full((256, 256, 3), [164.76139251,  167.47864617,  181.13838569])
        cv2.imwrite('%s/masked%d.png'%(args.outf, j), test_image)
        imagedata = torch.FloatTensor(cv2.resize(test_image, None, 
                 fx=scales[I], fy=scales[I], interpolation=cv2.INTER_LINEAR).transpose(2,0,1)).unsqueeze(0)

        #------------------------------------
        # Varied scale test the for last iter.
        if I == 2:
            test_data = torch.FloatTensor(cv2.resize(test_image,
                                         None, fx=2, fy=2).transpose(2,0,1)).unsqueeze(0)
            kern_N = illus2vec(test_data.cuda())
            simnet_N = Similarity(kern_N.detach(), chnls=chnls, sizes=(test_data.shape[2]*1), top_n=20).eval().cuda()
            
        #------------------------------------

        #print(tdataset.data[j], imagedata.shape)
        ret = ret_s[I]
        rois_s = []
        scores_s = [] 
        #print(rois_s)
        kern = illus2vec(imagedata.cuda()).cpu().numpy()
        summation = np.sum(kern, axis=1, keepdims=True)
        #print(summation)
        cv2.imwrite('summation.jpg', summation[0][0])
        kern = torch.FloatTensor(kern[0]).unsqueeze(0).cuda()
        
        if is_model:
            kern = featureEnc(kern)
        #print(kern.shape)#[:, :chnls, :, :]
        
        simnet = Similarity(kern.detach(), chnls=chnls, sizes=(imagedata.shape[2]*1), top_n=ret).eval().cuda()

        if I >= 1:
            dataset = DataDataset(args.dataset, base=base, chnls=chnls, mode='img', name=folders[I], file_list=file_list[I-1])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize//(4*I),
                                         shuffle=False, num_workers=1)
        else:
            #dataset = origindataset
            dataloader = torch.utils.data.DataLoader(origindataset, batch_size=args.batchSize,
                                         shuffle=False, num_workers=1)

        pbar = tqdm(total=len(dataloader))
        image = images[I]
        for i, data in enumerate(dataloader):
            data = illus2vec(data.cuda())
            #print('data:', data.shape)
            proposals, scores = simnet(data.cuda(), image[:data.shape[0],:,:,:])
            rois_s += proposals
            scores_s += scores
            pbar.update(1)
        pbar.close()

        if I==2:
            for i, data in enumerate(dataloader):
                data = illus2vec(data.cuda())
                proposals, scores = simnet_N(data.cuda(), image[:data.shape[0],:,:,:])
                rois_s += proposals
                scores_s += scores

        scores_ns = torch.cat(scores_s, 0)
        rois_ns = torch.cat(rois_s, 0).int()
        label = []
        for i, scores in enumerate(scores_s):
            label += [torch.ones_like(scores) * i]
        labels = torch.cat(label, 0)
        print(scores_ns.max())

        nscores, order = torch.sort(scores_ns, 0, True)
        '''
        for k in range(ret):
            idx = int(labels[order[k]])
            if I!=0:
                idx = file_list[I-1][idx%len(file_list[I-1])]
            file_list[I].append(idx)
        #print(file_list[I])
        file_list[I] = (np.unique(file_list[I])).tolist()
        #print(file_list[I])
        '''

    results = []
    for k in range(ret):
        idx = int(labels[order[k]])
        #idx = file_list[I-1][idx%len(file_list[I-1])]
        #print(idx)
        img = cv2.imread(origindataset.data[idx%20939].replace('simline', 'img').replace('.png', '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        x1,y1,x2,y2 = rois_ns[order[k]]
        cropped=img[max(0,y1):min(1170,y2), max(0,x1):min(827,x2),:]
        results.append(img[y1:y2, x1:x2,:])
        cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0), 3)
        if simline:
            cv2.imwrite('%s/Snormalize%d%s_%d.png'%(args.outf, j, tdataset.data[j].split('/')[-1],k), img)
        else:
            cv2.imwrite('%s/normalize%d%s_%d.png'%(args.outf, j, tdataset.data[j].split('/')[-1],k), img)

    fig = plt.figure(figsize=(13, 18))
    columns = 5
    rows = 8
    # ax enables access to manipulate each of subplots
    ax = []
    for i in range(len(results)):
        img = results[i]
        sim = nscores[i]
        # create subplot and append to ax
        t = fig.add_subplot(rows, columns, i+1)
        t.title.set_text('%.06f'%sim)
        t.set_axis_off()
        ax.append(t)
        plt.imshow(img)
    if simline:
        plt.savefig('%s/ConvSNM%d%s.png'%(args.outf, j, tdataset.data[j].split('/')[-1]), bbox_inches='tight', pad_inches=0.02)
    else:
        plt.savefig('%s/Conv%d%s.png'%(args.outf, j, tdataset.data[j].split('/')[-1]), bbox_inches='tight', pad_inches=0.02)