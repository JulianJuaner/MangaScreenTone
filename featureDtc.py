import cv2
import numpy as np
import sys
import os
import math

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from skimage.filters import gabor
from basicImgPcs import *

MEMORYLMT = True
SHOWIMG = True

def ShowAndWait(name, img):
    if SHOWIMG == False:
        return
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def GaborFilter():
    gabor = []
    for i in range(0, 6):
        gabor.append([])
        for j in range(0, 7):
            gaborFilter = cv2.getGaborKernel((10, 10), 15, np.deg2rad(i*15), (j+1)*5, 1) 
            display = (127*(gaborFilter / np.max(gaborFilter)) + 127).astype(np.uint8)
            gabor[i].append(gaborFilter)
    return gabor

def GaborFilter2():
    gabor = []
    scale = 4
    orientation=6
    minf = 0.1
    maxf = 0.3
    step = (maxf-minf)/(scale-1)
    print(step)
    base = maxf/minf
    a = math.pow(base,1.0/(scale-1))
    for i in range(orientation):
        print(i)
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
            # print(std, math.pi*i/4, 2*math.pi/f)
            # gaborFilter = cv2.getGaborKernel((20, 20), 15, np.deg2rad(i*15), (j+1)*5, 1) 
            gaborFilter = cv2.getGaborKernel((8, 8), std, math.pi*i/orientation, 2.0*math.pi/f, 1, 0) 
            # ShowAndWait("kernel", gaborFilter)
            # display = (127*(gaborFilter / np.max(gaborFilter)) + 127).astype(np.uint8)
            gabor.append(gaborFilter)
            ShowAndWait("kernel", gaborFilter)
    return gabor

def loadImages(folder):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            matches.append(os.path.join(root, filename))
   
    return matches

#erase line in the 
def EraseLine(img, Lineimg):
    erosion = Erosion(Lineimg, 2, IMG)
    mask_img = cv2.inRange(erosion, (0), (10))
    erosion = cv2.add(img, mask_img)
    return erosion

def GaborImagenary(sigma, theta, Lambda, psi, gamma):
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

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

def GaborDisplay(img, gabor):
    features = np.resize(cv2.filter2D(img, -1, gabor),\
                 (1, img.shape[0]*img.shape[1]))
    return features

def Save(img, path, folder):
    filename = os.path.basename(path)
    cv2.imwrite(os.path.join(folder, filename), img)

if __name__ == "__main__":
    imgPaths = loadImages('./label/imgs/')
    linePaths = loadImages('./label/line/')
    print("load img path: ", imgPaths)
    Gabor = GaborFilter()
    img = GetImg(imgPaths[0], PATH)
    filt_real, filt_imag = gabor(img, frequency=0.6)
    ShowAndWait("imaginary part", filt_real)
    #print(GaborImagenary(sigma, theta, Lambda, psi, gamma))
    '''
    for i in range(0, len(imgPaths)):
        img = GetImg(imgPaths[i], PATH)
        lineImg =cv2.resize(GetImg(linePaths[i], PATH), img.shape)
        if MEMORYLMT:
            img = cv2.resize(img, (500, 750))
            lineImg = cv2.resize(lineImg, (500, 750))
        
        img = EraseLine(img, lineImg)

        result = np.empty((1, img.shape[0]*img.shape[1]))
        for kernal in range(0, len(gabor)*len(gabor[0])):
            #print(result.shape)
            result = np.append(result, GaborDisplay(img, gabor[int(kernal/len(gabor[0]))]\
                     [kernal%len(gabor[0])]), axis = 0)
        pca = PCA(n_components=3)
        pca.fit(result)
        resultPic = pca.components_.copy()
        #resultPic = 255*normalize(resultPic)
        resultPic = np.transpose(resultPic)
        resultPic = np.resize(resultPic, (img.shape[0], img.shape[1], 3))
        resultPic = cv2.normalize(resultPic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        Save(resultPic, imgPaths[i], "./label/gaborfeature")
        ShowAndWait("result", resultPic)
    '''
