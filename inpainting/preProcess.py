from utils import *
import os
from tqdm import tqdm
from config import *

getdir = './data0/line/'
line_dir = './data0/line/'
npy_dir = './data0/seg/2/'
screentone_dir = './data0/imgs/'
already = 0

if GABOR:
    getdir = '../../PCAresult/'


if __name__ == "__main__":
    clock('start')
    
    #print(cv2.getBuildInformation())

    '''name_matches = loadImages(npy_dir)
    for step, (filename) in enumerate(tqdm(name_matches)):
        numpy_array = np.load(filename)
        ShowAndWait(numpy_array)'''

    name_matches = loadImages(getdir)
    line_matches = loadImages(line_dir)
    img_matches = loadImages(screentone_dir)
    file_iter = tqdm(name_matches)

    name_matches.sort()
    line_matches.sort()
    img_matches.sort()

    for step, (filename) in enumerate(file_iter):
        if step < already:
            continue
        filled_img = GetImg(filename, PATH_COLOR)
        if GABOR:
            lineImg = cv2.resize(GetImg(line_matches[step], PATH_COLOR),
                                 filled_img.shape[0:2][::-1])
            _,lineImg = cv2.threshold(lineImg,220,255,cv2.THRESH_BINARY)
            filled_img[lineImg<0.1] = 0
            cv2.imwrite(OUTPUT + "{:04d}.png".format(step), filled_img)
            mask_img =  cv2.resize((255 - GetImg(line_matches[step], PATH_GRAY).astype(float))/255,
                                    filled_img.shape[0:2][::-1])

        if GABOR  == False:
            print(img_matches[step])
            cv2.imwrite(SCTDIR + "{:04d}.png".format(step), GetImg(img_matches[step], PATH_GRAY))
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


