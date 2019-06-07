from utils import *
import os
from tqdm import tqdm
from config import *

getdir = './data0/line/'
line_dir = './data0/line/'
npy_dir = './data0/seg/2/'
already = 377

if GABOR:
    getdir = '../../PCA/PCAresult/'

if __name__ == "__main__":
    clock('start')
    
    #print(cv2.getBuildInformation())

    '''name_matches = loadImages(npy_dir)
    for step, (filename) in enumerate(tqdm(name_matches)):
        numpy_array = np.load(filename)
        ShowAndWait(numpy_array)'''

    name_matches = loadImages(getdir)
    line_matches = loadImages(line_dir)
    file_iter = tqdm(name_matches)

    for step, (filename) in enumerate(file_iter):
        if step < already:
            continue
        filled_img = GetImg(filename, PATH_COLOR)
        if GABOR:
            lineImg = GetImg(line_matches[step], PATH_COLOR)
            _,lineImg = cv2.threshold(lineImg,220,255,cv2.THRESH_BINARY)
            filled_img[lineImg<0.1] = 0
            cv2.imwrite(OUTPUT + "{:04d}.png".format(step), filled_img)

        if GABOR  == False:
            cv2.imwrite(ROOTDIR + "{:04d}.png".format(step), filled_img)
            _,filled_img = cv2.threshold(filled_img,220,255,cv2.THRESH_BINARY)
            filled_img = FloodFill(filled_img, 8, IMG)
            cv2.imwrite(OUTPUT + "{:04d}.png".format(step), filled_img)


        mask_img = (255 - GetImg(filename, PATH_GRAY).astype(float))/255
        mask_img = Dilation(mask_img, 7, IMG)
        filled_img[mask_img>0.1] = 255
        mask_img[mask_img>0.1] = 1
        cv2.imwrite(INPUT + "{:04d}.png".format(step), filled_img)

        if GABOR == False:
            cv2.imwrite(MASKDIR + "{:04d}.png".format(step), mask_img*255)
        #file_iter.update()

    clock('image pre-process:')


