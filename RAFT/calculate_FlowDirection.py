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

def main(img1,img2,uv,uv_gt,idx):
    image = str(img1)
    flow_png = str(img2)
    flow_uv = np.load(uv)
    flow_uv_gt= np.load(uv_gt)
    
    flow_uv_gt = flow_uv_gt.transpose(1,2,0)
    width, height, _ = flow_uv.shape

    print('\n shape uv:',flow_uv.shape)
    print('\n shape uv_gt:',flow_uv_gt.shape)
   

    window_name = os.path.basename(image) 

    im = cv.imread(image)
    fl_png = cv.imread(flow_png)
    color = (0, 0, 255) 
    color_gt= (255,0,0)
    color_centre=(255,255,255)
    color_centre_gt=(0,0,0)
    thickness = 2
    u, v,norm = FlowDirection(flow_uv, width, height)
    u_gt, v_gt,norm_gt = FlowDirection(flow_uv_gt, width, height)
   
    k1=10             # fattore moltiplicativo per la visualizzazione del flusso centrale
    k2=10            # fattore moltiplicativo per la visualizzazione del flusso nelle sottozone
    image = cv.arrowedLine(im, (int(height/2), int(width/2)), (int(height/2) + k1*round(u/norm), int(width/2) + k1*round(v/norm)), color_centre, thickness)
    image_flow = cv.arrowedLine(fl_png, (int(height/2), int(width/2)), (int(height/2) + k1*round(u/norm), int(width/2) + k1*round(v/norm)), color_centre, thickness)

    image = cv.arrowedLine(image, (int(height/2), int(width/2)), (int(height/2) + k1*round(u_gt/norm_gt), int(width/2) + k1*round(v_gt/norm_gt)), color_centre_gt, thickness)
    image_flow = cv.arrowedLine(image_flow, (int(height/2), int(width/2)), (int(height/2) + k1*round(u_gt/norm_gt), int(width/2) + k1*round(v_gt/norm_gt)), color_centre_gt, thickness)
    sections = 4

    for i in range(int(sections/2)):
        for j in range(int(sections/2)):

            flow = flow_uv[(i*int(width/2)):(i+1)*int(width/2), (j*int(height/2)):(j+1)*int(height/2), :] 
            u, v,norm = FlowDirection(flow, int(width/2), int(height/2))

            flow_gt = flow_uv_gt[(i*int(width/2)):(i+1)*int(width/2), (j*int(height/2)):(j+1)*int(height/2), :] 
            u_gt, v_gt,norm_gt = FlowDirection(flow_gt, int(width/2), int(height/2))
           # print('\n u',u)
            #norm= max(norm,norm_i)
            #print(' \n norm di u{}'.format(i),norm)
            start_point = (int(i*height/2 + height/4), int(j*width/2 + width/4))
            end_point = (start_point[0] + k2*round(u/norm), start_point[1] + k2*round(v/norm))

            end_point_gt = (start_point[0] + k2*round(u_gt/norm_gt), start_point[1] + k2*round(v_gt/norm_gt))

            image_flow = cv.arrowedLine(image_flow, start_point, end_point, color, thickness)
            image = cv.arrowedLine(image, start_point, end_point, color, thickness)
            image_flow = cv.arrowedLine(image_flow, start_point, end_point_gt, color_gt, thickness)
            image = cv.arrowedLine(image, start_point, end_point_gt, color_gt, thickness)
            # cv.imshow(window_name+'flow', image_flow) 
            # cv.waitKey(0)

    cv.imwrite('flow_vectors_no_mask_con_gt/image{}.png'.format(idx),np.concatenate([image, image_flow], axis=1))
    # cv.imshow(window_name, image) 
    # cv.imshow(window_name+'flow', image_flow) 
    # cv.waitKey(0)
    # cv.destroyWindow(window_name)
    # cv.destroyWindow(window_name+'flow')


if __name__ == "__main__":

    dir_image=Path(str(sys.argv[1]))
    dir_image_flow=Path(str(sys.argv[2]))
    dir_uv=Path(str(sys.argv[3]))
    dir_gt=Path(str(sys.argv[4]))
    for i in range(len(os.listdir(dir_image))):
        
        imfile1= os.path.join(dir_image,'frame{}_0.jpg'.format(i))
        imfile2= os.path.join(dir_image_flow,'frame{}_0.png'.format(i))
        imfile3= os.path.join(dir_uv,'Flow_inf{}.npy'.format(i))
        imfile4= os.path.join(dir_gt,'Flow_gt{}.npy'.format(i))
        

        if re.findall(r'\d+', str(imfile1))[0] != re.findall(r'\d+', str(imfile2))[0] != re.findall(r'\d+', str(imfile3))[0]:
            print('ERRORE COPPIA DI FRAME SBAGLIATA')
            break

        
        main(imfile1,imfile2,imfile3,imfile4,re.findall(r'\d+', str(imfile1))[0])
        




    