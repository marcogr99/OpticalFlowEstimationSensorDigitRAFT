from PIL import Image, ImageEnhance,ImageFilter
import numpy as np
import sys
import cv2 as cv
import os 

flowDir = sys.argv[1]
lst_img = os.listdir(flowDir)
aux = lst_img[:]
for file in aux:
    if 'Zone' in file:
        lst_img.remove(file)

for imgs in lst_img:

    img= Image.open(flowDir+imgs)
    converter= ImageEnhance.Color(img)
    enhancer= ImageEnhance.Contrast(img)
    img2= converter.enhance(1.5)
    enhancer2= ImageEnhance.Contrast(img2)
    img3=enhancer2.enhance(1.5)

    img3.save(imgs)

