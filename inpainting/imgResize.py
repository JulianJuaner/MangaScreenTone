import cv2
import os
import sys
import argparse
import time
from multiprocessing import Pool as ThreadPool

parser = argparse.ArgumentParser()
parser.add_argument('--line', type=str, default='./line', help='line root folder')
parser.add_argument('--img', type=str, default='./imgs', help='img root folder')
parser.add_argument('--out', type=str, default='./out', help='output folder')
args = parser.parse_args()

def loadImages(folder):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            matches.append(os.path.join(root, filename))
    return matches

def SameDirTree(in_dir, out_dir):
    matches = []
    outmatches = []
    for root, dirnames, filenames in os.walk(in_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            newpath = os.path.join(root, filename).replace(in_dir, out_dir)
            matches.append(filepath)
            outmatches.append(newpath)

    return matches, outmatches

line_matches = loadImages(args.line)
#print(line_matches)
img_matches, out_matches= SameDirTree(args.img, args.out)

def resizeImg(i):
    if img_matches[i].replace(args.img, args.line) not in line_matches:
        return
    img = cv2.imread(img_matches[i])
    if max(img.shape[0], img.shape[1]) < 1500:
        return
    line = cv2.imread(line_matches[line_matches.index(\
                    img_matches[i].replace(args.img, args.line))])

    img = cv2.resize(img, (line.shape[1], line.shape[0]))
    folder_path = out_matches[i].replace(os.path.basename(out_matches[i]), '')
    try:
        os.makedirs(folder_path)
    except:
        pass
    cv2.imwrite(out_matches[i], img)

#for i in range(len(line_matches)):
#   resizeImg(i)
start = time.time()
pool = ThreadPool(10)
pool.map(resizeImg, range(0, len(line_matches)))
print(time.time()-start, "time used")

