from PIL import Image, ImageDraw
import manga109api
import os
import time
import cv2
import numpy as np
from utils import *
from pprint import pprint

manga109_root_dir = "../../"
render_dir = "../../data/manga/train/trainA"

IMG = 0
PATH_COLOR = -1
PATH_GRAY = 1
start_time = 0
'''
def GetImg(imgPath, mode):
    if mode == IMG:
        img = imgPath
    if mode == PATH_COLOR:
        img = cv2.imread(imgPath)
    if mode == PATH_GRAY:
        img = cv2.imread(imgPath, 0)
    return img

def Dilation(imgPath, kernalSize=2, mode=IMG):
    img = GetImg(imgPath, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernalSize, kernalSize))
    ret,erosion = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    erosion = cv2.dilate(erosion, kernel, iterations=1)
    _, erosion = cv2.threshold(erosion,127,255,cv2.THRESH_BINARY)
    return erosion

def Erosion(imgPath, kernalSize=2, mode=IMG):
    img = GetImg(imgPath, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernalSize, kernalSize))
    ret,erosion = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    erosion = cv2.erode(erosion, kernel, iterations=1)
    _, erosion = cv2.threshold(erosion,127,255,cv2.THRESH_BINARY)
    return erosion'''
    
def draw_rectangle(img, x0, y0, x1, y1, annotation_type):
    assert annotation_type in ["body", "face", "frame", "text"]
    color = {"body": "#258039", "face": "#f5be41",
             "frame": "#31a9b8", "text": "#cf3721"}[annotation_type]
    width = 10
    draw = ImageDraw.Draw(img)
    draw.line([x0 - width/2, y0, x1 + width/2, y0], fill=color, width=width)
    draw.line([x1, y0, x1, y1], fill=color, width=width)
    draw.line([x1 + width/2, y1, x0 - width/2, y1], fill=color, width=width)
    draw.line([x0, y1, x0, y0], fill=color, width=width)

def loadImages(folder):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            #print(filename[-4:])
            if filename[-4:] in ['.jpg', '.png']:
                matches.append(os.path.join(root, filename))
    #print(matches)
    return matches

def train_save(folder):
    os.mkdir('../../../data/manga/test/lineA/')
    os.mkdir('../../../data/manga/train/lineA/')
    matches = loadImages(folder)
    for counter in range(len(matches)):
        img = cv2.imread(matches[counter])
        if counter%17 == 0:
            outf = '../../../data/manga/test/lineA/'
        else:
            outf = '../../../data/manga/train/lineA/'
        cv2.imwrite((outf +'{:05d}.jpg'.format(counter)), img)
    print('over.')

def mask_image(folder, outf):
    matches = loadImages(folder)
    for counter in range(len(matches)):
        path, name = os.path.split(matches[counter])
        img = GetImg(matches[counter], PATH_GRAY)
        img = Threshold(img, threshold=230)
        #erosion = Erosion(np.uint8(img), kernalSize=1)
        cv2.imwrite(outf + name, img)
    print('over')

def crop_save(folder):

    try:
        os.makedirs(folder + 'train/B/')
        os.makedirs(folder + 'test/B/')
        os.makedirs(folder + 'train/A/')
        os.makedirs(folder + 'test/A/')
    except:
        pass

    counter = 0
    print('start prasing...')
    p = manga109api.Parser(root_dir=manga109_root_dir)
    print('end')
    for m in range(0, len(p.books)):
        print(p.books[m])
        for index in range(1, len(p.annotations[p.books[m]]['book']['pages']['page']) - 1):
            #print(p.books[m], index)
            try:
                imgright = Image.open(p.img_path(book=p.books[m], index=2*index))
                imgleft = Image.open(p.img_path(book=p.books[m], index=2*index+1))
            except:
                continue

            width = imgleft.size[0]
            total_width = imgleft.size[0] + imgright.size[0]
            max_height = imgleft.size[1]
            new_im = Image.new('RGB', (total_width, max_height))
            new_im.paste(imgleft, (0,0))
            new_im.paste(imgright, (width,0))
            annotation_type = "frame"
            try:
                rois = p.annotations[p.books[m]]["book"]["pages"]["page"][index][annotation_type]
            except:
                continue
            #print(rois)
            if type(rois) is not list:
                rois = [rois]
            for roi in rois:
                #print(roi)
                xmin = roi["@xmin"]
                ymin = roi["@ymin"]
                xmax = roi["@xmax"]
                ymax = roi["@ymax"]
                width = (xmax-xmin)
                height = (ymax-ymin)
                ratio = width/height
                #print(width, height, ratio)
                if width < 768 and width>256 and height>256 and height<768 and (0.8<ratio and ratio<1.25):
                    #draw_rectangle(new_im,xmin,ymin,xmax,ymax, annotation_type)
                    #new_im.show()
                    counter+=1
                    box = (xmin+10, ymin+10, xmax-20, ymax-20)
                    cropped_image = new_im.crop(box)
                    #cropped_image.show()
                    if width < height:
                        cropped_image = cropped_image.resize((256, int(256*height/width)))
                    else:
                        cropped_image = cropped_image.resize((int(256*width/height), 256))

                    if counter%10==0:
                        outf = folder + 'test/B/'
                    else:
                        outf = folder + 'train/B/'
                    result = np.array(cropped_image)
                    cv2.imwrite(outf + "{:05d}.jpg".format(counter), result)

                        
            #new_im.show()
            #time.sleep(5)
def threshold(folder):
    os.mkdir('../../data/manga/test/newB/')
    os.mkdir('../../data/manga/train/newB/')

    matches = loadImages(folder)
    for counter in range(len(matches)):
        img = Threshold(cv2.imread(matches[counter]))
        if counter%17 == 0:
            outf = '../../data/manga/test/newB/'
        else:
            outf = '../../data/manga/train/newB/'
        cv2.imwrite((outf +'{:05d}.jpg'.format(counter)), img)
    print('over.')
#crop_save(manga109_root_dir)
#train_save(render_dir)
#threshold('../../data/manga/train/B/')
#os.mkdir('../../data/manga/train/A_mask/')
#os.mkdir('../../data/manga/test/A_mask/')
#os.mkdir('../../data/manga/train/B_mask3/')
#os.mkdir('../../data/manga/test/B_mask3/')

#mask_image('../../data/manga/train/A/', '../../data/manga/train/A_mask2/')
#mask_image('../../data/manga/test/A/', '../../data/manga/test/A_mask2/')
mask_image('../../data/manga/train/B/', '../../data/manga/train/B_mask/')
mask_image('../../data/manga/test/B/', '../../data/manga/test/B_mask/')