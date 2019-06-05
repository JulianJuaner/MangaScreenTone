import cv2
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt
from random import randint

ROOTDIR = './data/rawdata/'
INPUT = './data/input/'
OUTPUT = './data/valid/'
MASKDIR = './data/mask/'

COLORLEVELS = ["FF7700", "FFAAAA", "99CCCC", "FF9999", "996699", "0099CC"]
PATH = 1
IMG  = 0

def lmt(img):
    for pixelVal in img:
        if pixelVal > 255: pixelVal = 255
        if pixelVal < 0: pixelVal = 0
    return img


def Hex2RGB(strInput):
    return tuple(int(strInput[i:i + 2], 16) for i in (0, 2, 4))

def GetImg(imgPath, mode):
    if mode == IMG:
        img = imgPath
    if mode == PATH:
        img = cv2.imread(imgPath, 0)
    return img

#Erosion function for cv2.
def Erosion(imgPath, kernalSize, mode):
    img = GetImg(imgPath, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernalSize, kernalSize))
    ret,erosion = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    erosion = cv2.erode(img, kernel, iterations=1)
    return erosion

def Dilation(imgPath, kernalSize, mode):
    img = GetImg(imgPath, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernalSize, kernalSize))
    ret,dilation = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    dilation = cv2.dilate(img, kernel, iterations=1)
    return dilation



#FloodFill function, define number of color values.
def FloodFill(imgPath, numOfLevel, mode):
    img = GetImg(imgPath, mode)
    height, width = img.shape[0:2]
    upper = (10, 10, 10)
    lower = (10, 10, 10)
    mask = np.zeros((height + 2, width + 2), np.uint8)
    count = 0
    for i in range(0, height):
        for j in range(0, width):
            # in cv2.floodFill(), the coordinate of seed point
            # is (x, y), follows Cartesian system.
            if np.array_equal(img[i][j], [255, 255, 255]):
                color = COLORLEVELS[randint(0, numOfLevel - 1)]
                count += 1
                cv2.floodFill(img, mask, (j, i), Hex2RGB(color), lower,
                              upper)
    return img, mask

#Fill in patterns by color regions.
def FillinPattern(imgPath, numOfLevel, mode):
    img = GetImg(imgPath, mode)
    for pattern in range(0, numOfLevel):
        fillPattern = cv2.imread(('./patterns/00' + str(pattern+1) + '.jpg'), 1)
        color = COLORLEVELS[pattern]
        if fillPattern.shape[0] > img.shape[0] and fillPattern.shape[1] > img.shape[1]:
            fillPattern = fillPattern[0: img.shape[0], 0 : img.shape[1], :]
        #print(fillPattern.shape, img.shape)
        temp_mask = cv2.inRange(img, Hex2RGB(color), Hex2RGB(color))
        temp_mask_inv = cv2.bitwise_not(temp_mask)
        img = cv2.add(cv2.bitwise_and(img, img, mask=temp_mask_inv)\
            , cv2.bitwise_and(fillPattern, fillPattern, mask=temp_mask))
    cv2.imshow("emm", img)
    #cv2.imshow("emm?", cv2.bitwise_and(fillPattern, fillPattern, mask=temp_mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def RectInsideTest(img, cnt, rect):
    #print(rect)
    for point in rect:
        #print("Pount: ", point)
        if cv2.pointPolygonTest(cnt, tuple(point[0]), True) < 0: 
            return False
    blank = np.zeros((img.shape[1],img.shape[0],3), np.uint8)
    blank_2 =  np.ones((img.shape[1],img.shape[0],3), np.uint8)
    cv2.drawContours(blank, rect, -1, (0,255,0), 1)
    mask_img = cv2.inRange(blank, (0,255,0), (0,255,0))
    cv2.drawContours(blank_2, cnt, -1, (0,255,0), 1)
    compare = cv2.bitwise_and(blank, blank_2)
    #print((compare > 10).sum())
    if (compare > 10).sum()>3:
        return False
    return True 

def gabor_fn(sigma, theta, Lambda, psi, gamma):
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

def ScanLineRect(img, cnt):
    Xaxis = cnt[:, :, 0]
    Yaxis = cnt[:, :, 1]
    Min = min(Xaxis)[0]
    Max = max(Xaxis)[0]
    print(Min, Max, len(cnt))
    MaxArea = 0
    A_found = False
    pointA = []
    pointB = []
    pointC = []
    pointD = []
    result = []
    for i in range(Min, Max):
        for j in range(len(cnt)):
            if cnt[j][0][0] == i and A_found == True:
                #if cnt[j][0][1] - i == 1 or cnt[j][0][1] - i == -1:
                   # continue
                pointB = cnt[j][0].tolist()
                A_found = False
                apogee = Max
                for k in range(len(cnt)):
                    get = False
                    if cnt[k][0][1] == pointA[1]:
                        get = True
                        pointD = cnt[k][0].tolist()
                        pointC = [pointD[0], pointB[1]]
                    if cnt[k][0][1] == pointB[1]:
                        get = True
                        pointC = cnt[k][0].tolist()
                        pointD = [pointC[0], pointA[1]]
                    if get == True and RectInsideTest(img, cnt, np.asarray([[pointA], [pointB], [pointC], [pointD]])):
                        Area = (pointA[0] - pointC[0])* (pointA[1] - pointC[1])
                        #print("!")
                        if MaxArea < Area:
                            MaxArea = Area
                            result = [np.asarray([pointA, pointB, pointC, pointD])]
            if cnt[j][0][0] == i and A_found == False:
                pointA = cnt[j][0].tolist()
                A_found = True
    return result
    print(cnt[:, :, 1])

def RegionDetect(imgPath, mode):
    img = GetImg(imgPath, mode)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
    thresh = np.pad(thresh, ((2, 2), (2, 2)), 'constant', constant_values=(255, 255))
    img = cv2.resize(img, (img.shape[1]+2, img.shape[0]+2))
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    newContour = []
    for contour in contours:
        if cv2.contourArea(contour)>350:
            epsilon = 0.3*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)
            newContour.append(contour)
    #RectInsideTest(img, newContour[4], newContour[4])
    #print(len(newContour))
    for i in range(0, len(newContour)):
        tempImg = img.copy()
        cv2.drawContours(tempImg, newContour, i, (0,255,0), cv2.FILLED)
        # min outline box.
        rect = cv2.minAreaRect(newContour[i])
        x,y,w,h = cv2.boundingRect(newContour[i])
        #print(rect, x, y, w, h)
        insideBox =[np.asarray([[x+w,y+h], [x,y+h],[x,y],[x+w,y]])]
        #rect = ((x+w/2,y+h/2),(x+w,y+h), 0)
        #cv2.rectangle(tempImg,(x,y),(x+w,y+h),(0,255,0),2)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print(insideBox, [box], max(newContour[i][:, :, 1]))
        cv2.drawContours(tempImg, [box], 0,(0,0,255),2)
        #insideBox = [box].copy()
        factor = 1
        m = ScanLineRect(img, newContour[i])
        print("result: ", m, insideBox)
        cv2.drawContours(tempImg, m, 0, (255,0,255),cv2.FILLED)
        '''while True:
            if factor > 300: 
                print("ai...")
                break
            if RectInsideTest(img, newContour[i], insideBox) == False:
                #print(insideBox[0])
                insideBox[0][0] = [x - 1 for x in insideBox[0][0]]
                insideBox[0][1] = [insideBox[0][1][0] + 1,\
                                  insideBox[0][1][1] - 1]
                #print(insideBox, factor)
                insideBox[0][3] = [insideBox[0][3][0] - 1,\
                                  insideBox[0][3][1] + 1]
                insideBox[0][2] = [x + 1 for x in insideBox[0][2]]
                factor += 2
            else:
                cv2.drawContours(tempImg, insideBox, 0, (255,0,255),1)
                break'''
        
        cv2.imshow("contour", tempImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":

    if len(sys.argv) > 1:
        imgName = sys.argv[1]
    else:
        imgName = "./img/test2.jpg"
    erosion = ErosionDilation(imgName, 2, PATH)
    RegionDetect(erosion, IMG)
    erosion, mask = FloodFill(erosion, 6, IMG)
    #FillinPattern(erosion, 6, IMG)
    #cv2.imshow("contour", contour)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
