from utils import *
import os
from tqdm import tqdm

if __name__ == "__main__":
    clock('start')
    name_matches = loadImages(ROOTDIR)

    for step, (filename) in enumerate(tqdm(name_matches)):
        filled_img = GetImg(filename, PATH_COLOR)

        _,filled_img = cv2.threshold(filled_img,220,255,cv2.THRESH_BINARY)
        filled_img = FloodFill(filled_img, 8, IMG)
        cv2.imwrite(OUTPUT + FileName(filename, '.png'), filled_img)

        mask_img = (255 - GetImg(filename, PATH_GRAY).astype(float))/255
        mask_img = Dilation(mask_img, 7, IMG)
        filled_img[mask_img>0.1] = 255
        mask_img[mask_img>0.1] = 1
        cv2.imwrite(INPUT + FileName(filename, '.png'), filled_img)
        cv2.imwrite(MASKDIR + FileName(filename, '.png'), mask_img*255)

    clock('image pre-process:')


