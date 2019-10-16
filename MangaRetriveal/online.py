from __future__ import print_function
import os
import cv2
import time
import argparse

import torch
import numpy as np

from datasets import DataDataset, Erosion
from torch import nn
from torchvision import models
from rpn import Similarity
from models.networks import Illust2vecNet, myVGG, MultiScale, MultiLayer
from vae.encoder import FeatureEncoder, VAE
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--outf', type=str, default='image/out',  help='output folder')
parser.add_argument('--dataset', type=str, default='../../../data/manga',  help='dataset mode')
parser.add_argument('--testimgs', type=str, default='image/test',  help='test images')
parser.add_argument('--scales', type=str, default='0.5,1,2',  help='scales of test images')
parser.add_argument('--start', type=int, default=300, help='start epoch of encoder')
parser.add_argument('--modelf', type=str, default='two_addition',  help='model folder')
parser.add_argument('--test_mode', type=str, default='Default',  help='output folder')
parser.add_argument('--normalize', type=bool, default=True,  help='feature normalization')
args = parser.parse_args()
device = torch.device("cuda")
torch.cuda.set_device(args.gpu)

chnls = 512

base = 2**4

h,w = 1170//base,827//base

if 'no' in args.test_mode:
	is_model = False
	chnls = 2048
else:
	is_model =  True
	chnls = 64

is_mask = False
mb_size = args.batchSize
scales=[float(s)*0.5 for s in args.scales.split(',')]
illus2vec = MultiLayer('models/illust2vec_tag_ver200_2.pth').eval().cuda()

if is_model:
	featureEnc = FeatureEncoder().eval().cuda()
	featureEnc.load_state_dict(torch.load('models/model_%d.pth' % (args.modelf, args.start-1)))

# The heap structure to retain 20 top score images.
class TopNHeap():
	def __init__(self, topn):
		self.num = topn
		self.least = 0
		self.least_order = 0
		self.feature = []
		self.index = []
		self.scores = []
		
	def update(self, features, j):
		features = torch.cat(features, 0)
		scores, orders = torch.max(features.view(features.shape[0], -1), dim=1)
		
		test = max(scores)
		if test < self.least:
			return
		for i in range(len(scores)):

			if orders[i] in self.index:
				continue

			if len(self.scores)<self.num:
				self.scores += [scores[i].item()]
				self.index += [j*args.batchSize + i]
				self.least = min(self.scores)
				self.least_order = self.scores.index(self.least)

			elif scores[i]>self.least:
				self.scores[self.least_order] = scores[i].item()
				self.index[self.least_order] = j*args.batchSize + i
				self.least = min(self.scores)
				self.least_order = self.scores.index(self.least)

if is_model:
	dataset = DataDataset(args.dataset, mode='feature', name=args.modelf)
else:
	dirName = 'simline'
	dataset = DataDataset(args.dataset, mode='img', name=dirName)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=False, num_workers=1)
tdataset = DataDataset(args.testimgs, name='simline',basic=1)
tdataloader = torch.utils.data.DataLoader(tdataset, batch_size=1,
                                         shuffle=False, num_workers=1)
print('Images in dataset: %s'%len(dataset))
feature = torch.FloatTensor(np.random.rand(mb_size, chnls, 73, 51)).cuda()
image = torch.FloatTensor(np.zeros((mb_size, 1, int(1170), int(827)))).cuda()
print("fake img generated")
nets = []
rois_s = []
scores_s = []

levels = [255-164.76139251,  255-167.47864617,  255-181.13838569]
for j,tdata in enumerate(tdataloader):
	
	print(tdataset.data[j])
	ret = 20
	topN = TopNHeap(ret)
	rois_s = []
	scores_s = [] 

	test_image = tdata[0].numpy().transpose(1,2,0)

	imagemask = Erosion(test_image, 30)
	if is_mask:
		for count1 in range(256):
			for count2 in range(256):
				if 255 in imagemask[count1][count2]:
					for level in range(3):
						imagemask[count1][count2][level] = levels[level]
		test_image = test_image - imagemask
	imagedata = torch.FloatTensor(test_image.transpose(2,0,1)).unsqueeze(0)
	kern = illus2vec(imagedata.cuda(), mode='test')
	
	if is_model:
		
		kern = featureEnc(kern)
	print('kernel shape:', kern.shape)

	simnet = Similarity(kern.detach(),chnls=chnls, sizes=(tdata.shape[2]*1), top_n=ret).eval().cuda()
	pbar = tqdm(total=len(dataloader))

	for i, data in enumerate(dataloader):
		
		if is_model == False:
			features = illus2vec(data.cuda(), mode='train')
			proposals, scores = simnet(features, image[:data[0].shape[0],:,:,:], first=True)
		else:
			proposals, scores = simnet(data[0].cuda(), image[:data[0].shape[0],:,:,:], first=True)
		topN.update(proposals, i)

		pbar.update(1)
	pbar.close()

	for counter in range(len(topN.index)):
		if is_model == False:
			features = illus2vec(dataset[topN.index[counter]].unsqueeze(0).cuda(), mode='train')
			proposals, scores = simnet(features, image[:1,:,:,:])
		else:
			proposals, scores = simnet(dataset[topN.index[counter]][0].unsqueeze(0).cuda(), image[:1,:,:,:])
		rois_s += proposals
		scores_s += scores

	scores_ns = torch.cat(scores_s, 0)
	rois_ns = torch.cat(rois_s, 0).int()
	label = []
	for i, scores in enumerate(scores_s):
		label += [torch.ones_like(scores) * i]
	labels = torch.cat(label, 0)
	nscores, order = torch.sort(scores_ns, 0, True)
	
	results=[]
	for k in range(ret):
		idx = int(labels[order[k]])
		img = cv2.cvtColor(cv2.imread(dataset.data[topN.index[idx]%len(dataset)].replace('simline', 'img').replace('png', 'jpg')), cv2.COLOR_BGR2RGB)
		x1,y1,x2,y2 = rois_ns[order[k]]

		cropped = img[max(0,y1):min(1170,y2), max(0,x1):min(827,x2),:]
		results.append(img[y1:y2, x1:x2,:])
		cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0), 3)

	fig = plt.figure(figsize=(13, 9))
	columns = 5
	rows = 4
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
	os.makedirs('%s/%s'%(args.outf, args.test_mode), exist_ok=True)
	plt.savefig('%s/%s/%s.png'%(args.outf, args.test_mode, tdataset.data[j].split('/')[-1]), bbox_inches='tight', pad_inches=0.02)
