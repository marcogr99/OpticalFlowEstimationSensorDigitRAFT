import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import re

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo,count):
    

    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    #np.save('inf_npy/flow.npy',flo)
    # map flow to rgb image
    np.save('matrici_u_v_inferenza/Flow_inf{}.npy'.format(count),flo)
    
    flo = flow_viz.flow_to_image(flo)

    #img_flo = np.concatenate([img, flo], axis=0)
    
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey(0)
    
    cv2.imshow('imag',flo)
    cv2.waitKey(0)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    count=0
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg')) + \
                 glob.glob(os.path.join(args.path, '*.ppm'))
        
        images = sorted(images)
        
        for imfile1, imfile2 in zip(images[:-1:2], images[1::2]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            print('\n numero im1:', re.findall(r'\d+', str(imfile1)) )
            
            print('\n numero im2:', re.findall(r'\d+', str(imfile2)))
            if re.findall(r'\d+', str(imfile1))[0] != re.findall(r'\d+', str(imfile2))[0]:
                print('ERRORE COPPIA DI FRAME SBAGLIATA')
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            count=int(re.findall(r'\d+', str(imfile1))[0])
            viz(image1, flow_up,count)
            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()


    demo(args)
