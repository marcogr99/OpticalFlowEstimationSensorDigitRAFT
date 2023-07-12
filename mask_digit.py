import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import re
import numpy as np
import cv2 as cv
import subprocess
import time
import numpy as np
import re

def main():

    
    # path 
    for i in range(492, 514): #indice frame da mascherare con stessa maschera
        for j in range(2):
            path = sys.argv[1]+'frame{}_{}.jpg'.format(i,j)
            print('\n path', path)
            image = cv.imread(path)
            h, w, _ = image.shape
            # inserire r se si vuole maschera rettangolare oppure t se triangolare
            if sys.argv[2]=='r': 

                ### MASCHERA RETTANGOLARE
                # Window name in which image is displayed
                window_name = 'Image'
                
                # Start coordinate, represents the top left corner of rectangle
                start_point = (0, 0)
                
                # Ending coordinate, represents the bottom right corner of rectangle
                end_point = (int(w/4), int(h))
                
                # Color in BGR
                color = (255,255,255) #White
                
                # Line thickness ---> is = -1 is full colored
                thickness = -1

                # Draw a white rectangle 
                image_masked = cv.rectangle(image, start_point, end_point, color, thickness)

                # start_point2 = (0, int(h/1.3))
                # end_point2 = (w, h)
                # image_masked = cv.rectangle(image_masked, start_point2, end_point2, color, thickness)
                
                # Displaying the image 
                cv.imshow(window_name, image_masked) 
                cv.waitKey(0)
                
                cv.imwrite('Digit dataset/ridges_mask/'+os.path.basename(path), image_masked)
            elif sys.argv[2]=='t': 
                ### MASCHERA TRIANGOLARE

                pt1 = (int(w/4), int(h))
                pt2 = (w, int(h/4))
                pt3 = (int(w), h)

                triangle_cnt = np.array( [pt1, pt2, pt3] )

                image_masked = cv.drawContours(image, [triangle_cnt], 0, (255, 255, 255), -1)

                pt1 = (int(w/1.6), 0)
                pt2 = (int(w), 0)
                pt3 = (w, int(h/1.6))

                triangle_cnt2 = np.array( [pt1, pt2, pt3] )

                image_masked = cv.drawContours(image_masked, [triangle_cnt2], 0, (255, 255, 255), -1)


                cv.imshow("image", image_masked)
                cv.waitKey(0)
                cv.imwrite('Digit dataset/ridges_mask/'+os.path.basename(path), image_masked)
if __name__ == "__main__":

    main()