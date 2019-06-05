import cv2
import numpy as np
import sys
import os
import math
import torch
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from skimage.filters import gabor
#from basicImgPcs import *

PATH = 1
IMG  = 0
MEMORYLMT = False
CUDA = True
TORCH_METHOD = True

def ShowAndWait(name, img):

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def GetImg(imgPath, mode):

    if mode == IMG:
        img = imgPath
    if mode == PATH:
        img = cv2.imread(imgPath, 0)
    return img

def gabor_fn(sigma, theta, Lambda, gamma, psi, size = "None"):

    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gbreal = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    gbimg = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.sin(2 * np.pi / Lambda * x_theta + psi)
    if size != "None":
        centerx = int((xmax + 1 - xmin)//2-(size//2))
        centery = int((ymax + 1 - ymin)//2-(size//2))
        gbreal = gbreal[centerx : centerx + size, centery : centery + size]
        gbimg = gbimg[centerx : centerx + size, centery : centery + size]
    return gbreal, gbimg

def normalizeImg(img):

    return cv2.normalize(np.absolute(img), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

def GaborFilter():

    Gabor = []
    scale = 6
    orientation = 8
    minf = 0.05
    maxf = 0.4
    step = (maxf-minf)/(scale-1)

    base = maxf/minf
    a = math.pow(base,1.0/(scale-1))

    for j in range(scale):
        for i in range(orientation):

            f = (j+1)*step+minf
            u0 = maxf/math.pow(a,scale-(j+1))
            Uvar = (a-1)*u0/((a+1)*np.sqrt(2*np.log(2.0)))
            z = -2*np.log(2)*(Uvar*Uvar)/u0
            Vvar = np.tan(math.pi/(2.0*orientation))*(u0+z)/np.sqrt(2*np.log(2.0)-z*z/(Uvar*Uvar))
            Xvar = 1/(2*math.pi*Uvar)
            Yvar = 1/(2*math.pi*Vvar)

            std = np.sqrt(Xvar*Xvar+Yvar*Yvar)
            t1 = math.cos(math.pi / orientation * (i))
            t2 = math.sin(math.pi / orientation * (i))
            real = np.zeros((13, 13))
            imagin = np.zeros((13, 13))
            side = 6

            for x in range(0, 13):
                for y in range(0, 13):

                    X = (float(x)-side)*t1 +(float(y)-side)*t2
                    Y = -(float(x)-side)*t2 +(float(y)-side)*t1

                    G = 1.0 / (2.0*math.pi*Xvar*Yvar)*math.pow(a, float(scale) - (j+1))\
                        *math.exp(-0.5*((X*X)/(Xvar*Xvar) + (Y*Y)/(Yvar*Yvar)))
                    real[x][y] = G * math.cos(2.0*math.pi*u0*X)
                    imagin[x][y] = G * math.sin(2.0*math.pi*u0*X)
            
            result = 1j*imagin
            result += real
            Gabor.append(result)

    return Gabor

def GaborFilterComplex(img):

    scale = 4
    orientation=6
    minf = 0.1
    maxf = 0.3
    step = (maxf-minf)/(scale-1)
    base = maxf/minf
    a = math.pow(base,1.0/(scale-1))
    imgs = []

    for i in range(2):
        for j in range(scale):
            f = j*step+minf
            print(j,f)
            u0 = maxf/math.pow(a,scale-j)
            Uvar = (a-1)*u0/((a+1)*np.sqrt(2*np.log(2.0)))
            z = -2*np.log(2)*(Uvar*Uvar)/u0
            Vvar = np.tan(math.pi/(2.0*orientation))*(u0+z)/np.sqrt(2*np.log(2.0)-z*z/(Uvar*Uvar))
            Xvar = 1/(2*math.pi*Uvar)
            Yvar = 1/(2*math.pi*Vvar)
            std = np.sqrt(Xvar*Xvar+Yvar*Yvar)
            real, imagin = gabor(img, frequency = f, \
            n_stds = 3, theta = math.pi*i/orientation, sigma_x=Xvar, sigma_y=Yvar)

            imgs.append(real.reshape(1,-1))
            imgs.append(imagin.reshape(1,-1))

    return imgs

def loadImages(folder):

    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            matches.append(os.path.join(root, filename))
    return matches

class GaborWavelet(torch.nn.Module):

    def __init__(self, gabor):

        super(GaborWavelet, self).__init__()
        if CUDA == False:
            kernel1 = torch.FloatTensor(gabor.real).unsqueeze(0).unsqueeze(0)
            kernel2 = torch.FloatTensor(gabor.imag).unsqueeze(0).unsqueeze(0)
        else:
            kernel1 = torch.cuda.FloatTensor(gabor.real).unsqueeze(0).unsqueeze(0)
            kernel2 = torch.cuda.FloatTensor(gabor.imag).unsqueeze(0).unsqueeze(0)

        self.weight1 = torch.nn.Parameter(data=kernel1, requires_grad=False)
        self.weight2 = torch.nn.Parameter(data=kernel2, requires_grad=False)

    def forward(self, img):

        if CUDA == False:
            x1 = torch.nn.functional.conv2d(torch.FloatTensor(img).unsqueeze(0).unsqueeze(0), self.weight1, padding=6)
            x2 = torch.nn.functional.conv2d(torch.FloatTensor(img).unsqueeze(0).unsqueeze(0), self.weight2, padding=6)
            features =  torch.Tensor.numpy(x1) + 1j*torch.Tensor.numpy(x2)
        else:
            x1 = torch.nn.functional.conv2d(torch.cuda.FloatTensor(img).unsqueeze(0).unsqueeze(0), self.weight1, padding=6)
            x2 = torch.nn.functional.conv2d(torch.cuda.FloatTensor(img).unsqueeze(0).unsqueeze(0), self.weight2, padding=6)
            features =  x1.cpu().numpy() + 1j* x2.cpu().numpy()

        x = np.absolute(features.reshape(1,-1)).astype(float)
        return x

# Another way to implement torch conv2d
def GaborDisplay(img, Gabor):
    
    features = cv2.filter2D(img, -1, Gabor.real)
    features_complex = cv2.filter2D(img, -1, Gabor.imag)
    features = features + 1j* features_complex
    features = np.absolute(features.reshape(1,-1)).astype(float)

    return features

if __name__ == "__main__":
    imgPaths = loadImages('./label/imgs/')
    imgPaths.sort()
    linePaths = loadImages('./label/line/')
    linePaths.sort()
    print(linePaths)
    Gabor = GaborFilter()
    kernel = np.ones((3,3),np.uint8)
    count = 0
    for path,lpath in zip(imgPaths, linePaths):
        start_time = time.time()
        result = []
        print(lpath)
        line = (255 - GetImg(lpath, PATH).astype(float))/255
 
        if MEMORYLMT:
            #if image crop:
            img = cv2.resize(GetImg(path, PATH), line.shape[:2][::-1], interpolation=cv2.INTER_LINEAR).astype(float)/255
            line = line[0: 500, 0: 700]
            img = img[0: 500, 0: 700]
        else:
            img = cv2.resize(GetImg(path, PATH), line.shape[:2][::-1], interpolation=cv2.INTER_LINEAR).astype(float)/255

        line = cv2.dilate(line, kernel, iterations=1)
        cv2.imwrite('out/%sline.png'%os.path.splitext(os.path.basename(path))[0], 255*line)
        img[line>0.1] = 1
        cv2.imwrite('out/%spattern.png'%os.path.splitext(os.path.basename(path))[0], 255*img)

        for kernal in range(0, len(Gabor)):
            if TORCH_METHOD:
                wave = GaborWavelet(Gabor[kernal])
                if CUDA == True:
                    wave.cuda()
                result.append(wave.forward(img))
            else:
                result.append(GaborDisplay(img, Gabor[kernal]))

        result = np.concatenate(result, axis=0)
        print('time for processing no.%d image: '%count, time.time() - start_time)

        pca = PCA(n_components=3)
        pca.fit(result)

        R = normalizeImg(pca.components_[0]).reshape(img.shape[0], img.shape[1])
        G = normalizeImg(pca.components_[1]).reshape(img.shape[0], img.shape[1])
        B = normalizeImg(pca.components_[2]).reshape(img.shape[0], img.shape[1])
        cv2.imwrite('out/R%spca.png'%os.path.splitext(os.path.basename(path))[0], normalizeImg(R))
        cv2.imwrite('out/G%spca.png'%os.path.splitext(os.path.basename(path))[0], normalizeImg(G))
        cv2.imwrite('out/B%spca.png'%os.path.splitext(os.path.basename(path))[0], normalizeImg(B))
        
        resultPic = np.transpose(normalizeImg(pca.components_.copy())).reshape((img.shape[0], img.shape[1], 3))
        print('time added for PCA no.%d image: '%count, time.time() - start_time)

        cv2.imwrite('out/%spca.png'%os.path.splitext(os.path.basename(path))[0], normalizeImg(resultPic))
        count += 1
