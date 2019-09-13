import cairosvg
import glob
import os
import xml.etree.ElementTree as et
import numpy as np
import cv2
from PIL import Image
import random
import io
from datasets import make_dataset

path = './test_unseen/'
save_path = './test_varywidth/'

color_pallate = ["#0000ff","#00ff00","#ff0000","#005bff","#5b00ff", "#5bff00","#00007f", "#ffa300", "#7fff7f", "#ff1200", "#c8ff36", "#0012ff", "#ff5b00", "#c80000", "#00ecff", "#ffec00","#ff00ec","#00a3ff", "#0000c8",  "#36ffc8", "#0096ff", "#ff9600", "#fff300", "#0000dc", "#00dc00","#ff3900", "#ff0039", "#00ff39"]


def read_svg(file_path):
	baseName = os.path.basename(file_path)
	baseName = baseName.split('.')[0]

	with open(file_path, 'r') as f:
		svg = f.read()
	
	svg_xml = et.fromstring(svg)
	i = 0
	for child in svg_xml:
		# w = random.randint(0, len(color_pallate)-1)
		w = i % len(color_pallate)
		i += 1
		# print(child.get('stroke'))
		child.set('stroke', color_pallate[w])
		child.set('stroke-width', '2')
		child.set('fill', 'none')
	# r = et.Element("rect")
	# r.set('width', '128')
	# r.set('height', '128')
	# r.set('height', '128')
	# r.set('style','fill:rgb(255,255,255)')
	# svg_xml.insert(1, r)
	# svg_out = et.tostring(svg_xml, method='xml')
	# png = cairosvg.svg2png(bytestring=svg_out)
	# out_img = Image.open(io.BytesIO(png))
	# out_img.save(save_path  + baseName + '.png')
	svg_out = et.tostring(svg_xml, encoding='unicode')
	f = open("noise/40_color.svg", "w")
	f.write(svg_out)
	f.close()


def change_width(file_path):
	baseName = os.path.basename(file_path)
	baseName = baseName.split('.')[0]

	with open(file_path, 'r') as f:
		svg = f.read()
	
	svg_xml = et.fromstring(svg)
	i = 0
	for child in svg_xml:
		child.set('stroke-width', '2.2')
		# child.set('fill', 'none')

	svg_out = et.tostring(svg_xml, encoding='unicode')
	f = open('../../../data/manga/modifysvg/'+baseName+".svg", "w")
	f.write(svg_out)
	f.close()
from multiprocessing import Pool as ThreadPool

def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (https://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (https://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (https://stackoverflow.com/a/8720632  and  https://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

def exportsvg(fromDir, targetDir, exportType='png'):

    num = 0
    for a,f,c in os.walk(fromDir):
        for fileName in c:
            path = os.path.join(a,fileName)
            if os.path.isfile(path) and fileName[-3:] == "svg":
                num += 1
                fileHandle = open(path)
                svg = fileHandle.read()
                fileHandle.close()
                exportPath = os.path.join(targetDir, fileName[:-3] + exportType)
                exportFileHandle = open(exportPath,'w')

                if exportType == "png":
                    try:
                        cairosvg.svg2png(bytestring=svg, write_to=exportPath)
                    except:
                        continue

                exportFileHandle.close()


#---------------------------------------
'''svgDir = '.'
exportDir = '.'
exportFormat = 'png'
if not os.path.exists(exportDir):
    os.mkdir(exportDir)
exportsvg(svgDir, exportDir, exportFormat)'''

def Erosion(img, kernalSize=2, mode=1):
    #img = GetImg(imgPath, mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernalSize, kernalSize))
    #ret,erosion = cv2.threshold(img,,255,cv2.THRESH_BINARY)
    erosion = cv2.erode(img, kernel, iterations=1)
    #_, erosion = cv2.threshold(erosion,127,255,cv2.THRESH_BINARY)
    return erosion

if __name__ == '__main__':
	#originlist = make_dataset('../../../data/manga/modifysvg')
	svglist = make_dataset('../../../data/manga/png')
	#print(len(svglist), len(originlist))

	'''print('stroke')
	for i in range(len(originlist)):
		change_width(originlist[i])'''
	print('export')
	#os.mkdir('../../../data/manga/png1')
	#exportsvg('../../../data/manga/modifysvg', '../../../data/manga/png1')
	#os.mkdir('../../../data/manga/train/lineA/')
	#svglist = make_dataset('../../../data/manga/png1')
	print('resize')
	for i in range(len(svglist)):
		if i%100==0:
			print(svglist[i])
		if i%17 == 0:
			outf = '../../../data/manga/test/lineA/'
		else:
			outf = '../../../data/manga/train/lineA/'
		new_img = Erosion(cv2.imread(svglist[i], 0))
		
		new_img = cv2.resize(new_img, (256,256), interpolation = cv2.INTER_NEAREST)
		print(new_img.shape)
		cv2.imwrite((outf +'{:05d}.jpg'.format(i)), new_img)
	#read_svg('noise/40.svg')
