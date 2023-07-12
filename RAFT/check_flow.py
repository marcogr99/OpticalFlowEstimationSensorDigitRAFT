import numpy as np
import cv2 as cv
import sys
import pandas as pd
import os.path
from pathlib import Path
import time
import torch
import os.path
import glob
import matplotlib.pyplot as plt
import csv
import math

def check_flow(flow_gt,flow_inf,threshold):
    name=os.path.basename(flow_gt)
    flow_gt = np.load(flow_gt)
    flow_gt = flow_gt.transpose(1,2,0)
    print('\n flow gt shape', flow_gt.shape)
    flow_inf = np.load(flow_inf)
    print ('flow inf shape', flow_inf.shape)
    threshold = int(threshold)

    if flow_gt.shape[2] != 2 :
        print('errore su ground truth: non è u e v')
        return 
    if flow_inf.shape[2] != 2 :
        print('errore su inference: non è u e v')
        return 
    
    if flow_inf.shape != flow_gt.shape :

        print('dimensioni matrice inf:',flow_inf.shape)
        print('dimensioni matrice gt:',flow_gt.shape)
        print('errore, dimensioni non coincidenti')
        return
    
    H=flow_gt.shape[0]
    W=flow_gt.shape[1]
    vettore_diff=[]
    count_acc=0
    vec_u = []
    vec_v = []
    EPE= False
    
    
    for i in range(0,H):
        for j in range(0,W):
            
            u_gt= flow_gt[i,j,0]
            u_inf= flow_inf[i,j,0]
            v_gt= flow_gt[i,j,1]
            v_inf= flow_inf[i,j,1]


            if EPE :
                norm_gt=np.linalg.norm(flow_gt[i,j,:])
                norm_inf=np.linalg.norm(flow_inf[i,j,:])
                diff_norm= 100*abs(norm_gt-norm_inf)/norm_gt   #  è il valore percentuale di quanto l'inferenza è sbagliata rispetto al ground-truth
            
            else:
               delta_u=u_inf - u_gt
               delta_v= v_inf - v_gt
               diff_norm= 100*math.sqrt(delta_u**2+delta_v**2)/min((u_gt**2+v_gt**2),(u_inf**2+v_inf**2))

            

            diff_u = abs(flow_gt[i,j,0]) - abs(flow_inf[i,j,0])
            diff_v = abs(flow_gt[i,j,1]) - abs(flow_inf[i,j,1])
            vec_u.append(diff_u)
            vec_v.append(diff_v)
           
            vettore_diff.append(diff_norm)

            if diff_norm < threshold:     
                count_acc= count_acc +1
                
    accuracy= count_acc/(H*W) 
    

    max_diff=max(vettore_diff)  
    min_diff = min(vettore_diff)
    max_diff_u = max(vec_u)
    max_diff_v = max(vec_v)

    range_u_gt = [min(map(min, (flow_gt[:,:,0]))), max(map(max, (flow_gt[:,:,0])))]
    range_u_inf = [min(map(min, (flow_inf[:,:,0]))), max(map(max, (flow_inf[:,:,0])))] 
    range_v_gt = [min(map(min, (flow_gt[:,:,1]))), max(map(max, (flow_gt[:,:,1])))]
    range_v_inf = [min(map(min, (flow_inf[:,:,1]))), max(map(max, (flow_inf[:,:,1])))]




    print('\n range di variazione di u in gt', range_u_gt)
    print('\n range di variazione di u in inf', range_u_inf)
    print('\n range di variazione di v in gt', range_v_gt)
    print('\n range di variazione di v in inf', range_v_inf) 
    print('\n Numero pixel entro soglia', count_acc)    
    print('\n accuracy:',accuracy)    
    print('\n max difference:',max_diff)
    print('\n min difference:',min_diff)
    print('\n max diff u', max_diff_u)
    print('\n max diff v', max_diff_v)
    file_save = open('Risultati_accuratezza.txt','a')

    file_save.write('\n \n {}: \n accuray: {}, \n range_u_gt: {}, \n range_v_gt: {}, \n range_u_inf: {}, \n range_v_inf: {}, \n threshold: {}'.format(name,accuracy, range_u_gt,range_v_gt,range_u_inf,range_v_inf,threshold))
    #file.writerow([name,accuracy, range_u_gt,range_v_gt,range_u_inf,range_v_inf,threshold])
    
    return accuracy

if __name__ == '__main__':
    '''
    Primo Argomento: Path to Flow groundtruth
    Secondo Argomento: Path to Flow inference
    Terzo Argomento: Threshold minima differenza tra pixel
    '''
    dir_gt=Path(str(sys.argv[1]))
    dir_inf=Path(str(sys.argv[2]))
    vect_accuracy=[]
    #file= open('../accuratezza.xlsx','w')
    #file=csv.writer(file)
    #file.writerow(['Frame','Accuracy','range_u_gt','range_v_gt','range_u_inf','range_v_inf','threshold %'])
    for i in range(len(os.listdir(dir_gt))):
        
        gt= os.path.join(dir_gt,'Flow_gt{}.npy'.format(i))
        inf= os.path.join(dir_inf,'Flow_inf{}.npy'.format(i))
        #check_flow(gt, inf, sys.argv[3],file)
        acc=check_flow(gt, inf, sys.argv[3])
        #print('acc:',acc)
        vect_accuracy.append(acc)
        #print('\n vect_accuracy :', vect_accuracy)

    print('media acc:',(np.mean(vect_accuracy)))
    file_save1 = open('media_accuratezza.txt','a')
    file_save1.write('\n \n accuratezza media {}'.format(np.mean(vect_accuracy)))
    



        

    


    
    



