import matplotlib.pyplot as plt
import sys
import cv2 as cv 
import os
import glob
import numpy as np
from PIL import Image
import re
from pathlib import Path 



def FlowDirection(flow_uv, width, height):
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    u_sum = 0
    v_sum = 0

    for i in range(width):
        for j in range(height):
            u_sum += u[i, j]
            v_sum += v[i, j]

    #norm = min(len(str(int(abs(u_sum)))), len(str(int(abs(v_sum))))) 
    #norm=np.linalg.norm([u_sum,v_sum])
    norm = width*height

    
    return u_sum, v_sum ,norm

def Flow_RealTime(img1,img2,uv):
    image = str(img1)
    flow_png = str(img2)
    flow_uv = uv
    flow_uv = np.load(flow_uv)
    #flow_uv = flow_uv.transpose(1,2,0)
    width, height, _ = flow_uv.shape
   

    window_name = os.path.basename(image) 

    im = cv.imread(image)
    fl_png = cv.imread(flow_png)
    color = (0, 0, 255) 
    color_centre=(255,255,255)
    thickness = 2
    u, v,norm = FlowDirection(flow_uv, width, height)
   
    k1=50             # fattore moltiplicativo per la visualizzazione del flusso centrale
    k2=50            # fattore moltiplicativo per la visualizzazione del flusso nelle sottozone
    print('\n image:',image)
    print('\n image:',image_flow)
    image = cv.arrowedLine(im, (int(height/2), int(width/2)), (int(height/2) + k1*round(u/norm), int(width/2) + k1*round(v/norm)), color_centre, thickness)
    image_flow = cv.arrowedLine(fl_png, (int(height/2), int(width/2)), (int(height/2) + k1*round(u/norm), int(width/2) + k1*round(v/norm)), color_centre, thickness)

    sections = 4

    for i in range(int(sections/2)):
        for j in range(int(sections/2)):

            flow = flow_uv[(i*int(width/2)):(i+1)*int(width/2), (j*int(height/2)):(j+1)*int(height/2), :] 
            u, v,norm = FlowDirection(flow, int(width/2), int(height/2))
           # print('\n u',u)
            #norm= max(norm,norm_i)
            #print(' \n norm di u{}'.format(i),norm)
            start_point = (int(i*height/2 + height/4), int(j*width/2 + width/4))
            end_point = (start_point[0] + k2*round(u/norm), start_point[1] + k2*round(v/norm))
            image_flow = cv.arrowedLine(image_flow, start_point, end_point, color, thickness)
            image = cv.arrowedLine(image, start_point, end_point, color, thickness)
            # cv.imshow(window_name+'flow', image_flow) 
            # cv.waitKey(0)
    cv.imwrite('flow_vectors_real_time/image.png',np.concatenate([image, image_flow], axis=1))
    
    cv.imshow(window_name, image) 
    #cv.imshow(window_name+'flow', image_flow) 
    cv.waitKey(200)

if __name__ == "__main__":

    
        
    imfile1= sys.argv[1]
    imfile2= sys.argv[2]
    imfile3= sys.argv[3]
    print('\n file 1:',imfile1)
    print('\n file 1:',imfile2)
    print('\n file 1:',imfile3)
    main(imfile1,imfile2,imfile3)
        




    