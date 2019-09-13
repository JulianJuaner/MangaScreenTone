from __future__ import print_function
import os
import cv2
import numpy as np
import torch
from models.networks import Illust2vecNet
from datasets import DataDataset, make_dataset
from vae.encoder import VAE
from sklearn.decomposition import PCA
import skimage
import time
import scipy.sparse
import argparse
#pca = PCA(n_components=5)
def two_addition(feature):
	#pathes = make_dataset('../../../../data/manga/featureHalfNC')
	result = np.zeros((256, feature.shape[1], feature.shape[2])).astype(np.float32)
	for j in range(256):
		result[j] = np.subtract(feature[2*j], feature[2*j+1])
	return result
		#np.save(os.path.join(manga_pth,
		#	name,dataset.data[i*mb_size+j].split('/')[-2],
		#	dataset.data[i*mb_size+j].split('/')[-1].replace('.png','.npy')), feat1[j])

def offline(args):
	#featureEnc = VAE().eval().cuda()
	#featureEnc.load_state_dict(torch.load('vae/checkpoints/%s/model_%d.pth' % (args.modelf, args.start-1)))
	torch.cuda.set_device(args.gpu)
	illus2vec = Illust2vecNet('models/illust2vec_tag_ver200_2.pth').cuda()
	illus2vec.eval()

	mb_size=args.batchSize
	manga_pth = args.dataset
	books = os.listdir(os.path.join(manga_pth,'image'))
	books.sort()
	name = 'FEATURE_POOL'
	start = time.time()
	for book in books:
		try:
			os.makedirs(os.path.join(manga_pth, name, book), exist_ok=True)
		except:
			pass
	dataset = DataDataset(manga_pth, base=0, name=name)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=mb_size,
											shuffle=False, num_workers=1)
	#Already = 5000
	
	print(len(dataset))
	for i,data in enumerate(dataloader):
		feat1 = illus2vec.forward(data.cuda()).cpu().numpy()
		for j in range(data.shape[0]):
			result = two_addition(feat1[j])
			if i%10000 == 0:
				print(data[j].shape)
			print(os.path.join(manga_pth,
			name,dataset.data[i*mb_size+j].split('/')[-2],
			dataset.data[i*mb_size+j].split('/')[-1]))
			torch.save(torch.FloatTensor(result), os.path.join(manga_pth,
			name,dataset.data[i*mb_size+j].split('/')[-2],
			dataset.data[i*mb_size+j].split('/')[-1].replace('.png','.pt')))
	'''
	print(len(dataset))
	for i,data in enumerate(dataloader):
		if i%100 == 0:
				print(i)
		#if i<=Already:
			#continue
		feat1 = illus2vec.forward(data.cuda()).cpu().numpy()
		feat1 = feature
		#_, feat1 = torch.max(illus2vec(data.cuda()), 1, keepdim=True)
		#feat1 = torch.div(feat1.float(), 255)
		#print(feat1.shape)
		for j in range(feat1.shape[0]):
			#result = two_addition(feat1[j])
			if i%10000 == 0:
				print(feat1[j].shape)
			np.save(os.path.join(manga_pth,
			name,dataset.data[i*mb_size+j].split('/')[-2],
			dataset.data[i*mb_size+j].split('/')[-1].replace('.png','.npy')), feat1[j])
	'''
	end = time.time()
	print(end-start)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
	parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
	parser.add_argument('--dataset', type=str, default='../../../data/manga',  help='dataset mode')
	parser.add_argument('--modelf', type=str, default='VAE_02',  help='model name')
	args = parser.parse_args()
	offline(args)