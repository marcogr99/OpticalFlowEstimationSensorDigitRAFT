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
    
    np.save('inf_npy/flow{}.npy'.format(count),flo)
    # map flow to rgb image
    #np.save('Flow_inf{}.npy'.format(count),flo)
    
    flo = flow_viz.flow_to_image(flo)

    img_flo = np.concatenate([img, flo], axis=0)
    
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey(0)
    
    cv2.imwrite('inf_images/flow{}.png'.format(count),flo)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    count=0
    with torch.no_grad():
        # images = glob.glob(os.path.join(args.path, '*.png')) + \
        #          glob.glob(os.path.join(args.path, '*.jpg')) + \
        #          glob.glob(os.path.join(args.path, '*.ppm'))
        
        # images = sorted(images)
        
        # # for imfile1, imfile2 in zip(images[:-1:2], images[1::2]):
        # #     image1 = load_image(imfile1)
        # #     image2 = load_image(imfile2)
            
            
        # #     padder = InputPadder(image1.shape)
        # #     image1, image2 = padder.pad(image1, image2)

        # #     flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        # #     count=int(re.findall(r'\d+', str(imfile1))[0])
        # #     viz(image1, flow_up,count)
        
        imfile1 = args.img1
        imfile2 = args.img2
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        print("\n            STO INFERENDO LE IMMAGINI:", imfile1, imfile2)
        
        
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        count=int(re.findall(r'\d+', str(imfile1))[0])
        #viz(image1, flow_up,count)
        
        flo = flow_up[0].permute(1,2,0).cpu().numpy()
    
        np.save('inf_npy/flow{}.npy'.format(count),flo)
            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    
    parser.add_argument('--img1', help="img1")
    parser.add_argument('--img2', help="img2")
    
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    #if not os.path.exists('inf_images'):

       # os.makedirs('inf_images')

    if not os.path.exists('inf_npy'):

        os.makedirs('inf_npy')

    demo(args)
