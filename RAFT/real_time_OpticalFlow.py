import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import re
import numpy as np
import cv2 as cv
import subprocess
import time
import math


def FlowDirection(flow_uv, width, height):

    
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    # u_sum = 0
    # v_sum = 0

    # for i in range(width):
    #     for j in range(height):
    #         u_sum += u[i, j]
    #         v_sum += v[i, j]
    # print('\n tempo for :', time.time() - start_for)
    
    v_sum=np.sum(v)
    u_sum=np.sum(u)
    
    
    #norm = min(len(str(int(abs(u_sum)))), len(str(int(abs(v_sum))))) 
    #norm=np.linalg.norm([u_sum,v_sum])
    norm = width*height

    
    return u_sum, v_sum ,norm

def Flow_RealTime(img1,uv):
    image = str(img1)
    #flow_png = str(img2)
    flow_uv = uv
    time.sleep(0.01)
    flow_uv = np.load(str(flow_uv))
    #flow_uv = flow_uv.transpose(1,2,0)
    width, height, _ = flow_uv.shape
   

    window_name = os.path.basename(image) 

    im = cv.imread(image)
    #fl_png = cv.imread(flow_png)
    color = (0, 0, 255) 
    color_centre=(255,255,255)
    thickness = 2
    u, v,norm = FlowDirection(flow_uv, width, height)
   
    k1=10            # fattore moltiplicativo per la visualizzazione del flusso centrale
    k2=10            # fattore moltiplicativo per la visualizzazione del flusso nelle sottozone

    #print(image)
    image = cv.arrowedLine(im, (int(height/2), int(width/2)), (int(height/2) + k1*round(u/norm), int(width/2) + k1*round(v/norm)), color_centre, thickness)
    #image_flow = cv.arrowedLine(fl_png, (int(height/2), int(width/2)), (int(height/2) + k1*round(u/norm), int(width/2) + k1*round(v/norm)), color_centre, thickness)
    
    # sections = 4

    # for i in range(int(sections/2)):
    #     for j in range(int(sections/2)):

    #         flow = flow_uv[(i*int(width/2)):(i+1)*int(width/2), (j*int(height/2)):(j+1)*int(height/2), :] 
    #         u, v,norm = FlowDirection(flow, int(width/2), int(height/2))
    #        # print('\n u',u)
    #         #norm= max(norm,norm_i)
    #         #print(' \n norm di u{}'.format(i),norm)
    #         start_point = (int(i*height/2 + height/4), int(j*width/2 + width/4))
    #         end_point = (start_point[0] + k2*round(u/norm), start_point[1] + k2*round(v/norm))
    #         #image_flow = cv.arrowedLine(image_flow, start_point, end_point, color, thickness)
    #         image = cv.arrowedLine(image, start_point, end_point, color, thickness)
    #         # cv.imshow(window_name+'flow', image_flow) 
    #         # cv.waitKey(0)
    #         #print('\n endpoint', end_point)
    # #cv.imwrite('flow_vectors_real_time/image.png',np.concatenate([image, image_flow], axis=1))
    
    # #cv.imshow(window_name, image) 
    # #cv.imshow(window_name+'flow', image_flow)
    #print('\n errore angolare :', math.atan2(u,v))

    return image

def main():
    if not os.path.exists("real_time_data"):
     
        os.makedirs("real_time_data")
    
    #Cattura video dal Digit
    vidcap = cv.VideoCapture('Dataset_digit_separati_per_tipo_tavola_video/Ridge3/R3_ort_2.mp4')
    fps = vidcap.get(cv.CAP_PROP_FPS)
    frame_interval = 4  # Estrai un frame ogni 30 frame int(fps/8+1)
    
    print('\n frame_interval :', frame_interval)
    
    # print('\n fps', fps)
    # print('\n frame interval', frame_interval)
    
    frame_list = []
    frame_count = 0
    n_salti = 0

    while True:
        #time.sleep(0.5)
        
        if vidcap.isOpened():
            ret, frame = vidcap.read()
            
            
            
        else:
            print("cannot open camera")
    


        if frame_count % frame_interval == 0:
                    
            
            frame_list.append(frame)
            #print('\n frame list:',frame_list)
            if len(frame_list)>=2 :
                    image1 = frame_list[-2]
                    image2 = frame_list[-1]

                    cv.imwrite("real_time_data/frame{}.png".format(len(frame_list)-2), image1)
                    cv.imwrite("real_time_data/frame{}.png".format(len(frame_list)-1), image2)
                    ######OCCHIO A QUANDO PRENDE I FRAME DA DENTRO DEMO.PY############
                    
                    process = subprocess.Popen(['python3','demo_real_time.py', '--model', 'checkpoints/raft-digit_07_07_fine_tuning.pth' , '--img1', "real_time_data/frame{}.png".format(len(frame_list)-2), '--img2', "real_time_data/frame{}.png".format(len(frame_list)-1)])
                    
                    #time.sleep(2)
                    npy = False
                    while not npy:
                        if os.path.isfile("inf_npy/flow{}.npy".format(len(frame_list)-2)):
                            npy = True
                        else:
                            print('Aspetto il npy venga salvato')
                            time.sleep(0.3)
                    #flow = Path("inf_images/flow{}.png".format(len(frame_list)-2))
                    #flow = Path("inf_images/flow{}.png".format(len(os.listdir("inf_images/"))-1))
                    
                    image1 = Path("real_time_data/frame{}.png".format(len(frame_list)-2))
                    flow_npy = Path("inf_npy/flow{}.npy".format(len(frame_list)-2))
                    #flow_npy = Path("inf_npy/flow{}.npy".format(len(os.listdir("inf_npy/"))-1))
                    
                    print('\n image 1', image1, os.path.isfile(image1))
                    #print('\n fl', flow, os.path.isfile(flow))
                    
                    print('\n npy', flow_npy, os.path.isfile(flow_npy))
                    if os.path.isfile(flow_npy) and os.path.isfile(image1):   #and os.path.isfile(flow):
                        start_time=time.time()
                        vectors = Flow_RealTime(image1,str(flow_npy))
                        print('\nMostro i frame:', '\n image1:',re.findall(r'\d+', str(image1))[0], '    flow npy:', re.findall(r'\d+', str(flow_npy))[0])
                        print('\nsecondi : ', (time.time()- start_time))
                        cv.imshow('arrows', vectors)
                        cv.waitKey(1)
                    else:
                        print('SALTO')
                        n_salti += 1
            
            #print('\n Frame count', frame_count)
            print('frame saltati', n_salti)
        frame_count += 1
        #print('\n frame_count',frame_count)
                        




if __name__ == "__main__":
    
    x = 2 #secondi da aspettare
    print('Il codice inizierà tra {} secondi'.format(x))

    time.sleep(x)
    main()






    


   