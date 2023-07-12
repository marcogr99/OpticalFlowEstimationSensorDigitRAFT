#import png
#import pfm
import numpy as np
#import matplotlib.colors as cl
#import matplotlib.pyplot as plt
#from PIL import Image
import sys
import cv2 as cv
import torch

#def read_flo_file(filename):
"""
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
"""
"""
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d
    """


def flow2rgb(flow_map, max_value):
    flow_map = torch.from_numpy(flow_map)
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Specify a .flo file on the command line.')
    else:
        with open(sys.argv[1], 'rb') as f:
            magic, = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
            else:
                w, h = np.fromfile(f, np.int32, count=2)
                print(f'Reading {w} x {h} flo file')
                data = np.fromfile(f, np.float32, count=2*w*h)
                # Reshape data into 3D array (columns, rows, bands)
                data2D = np.resize(data, (w, h, 2))
                print(data2D)
    
    flow_img = flow2rgb(data2D , None)
    
    flow_img1 = (flow_img*255).astype(np.uint8).transpose(1,2,0)
    
    cv.imshow('flow image',flow_img1)
    cv.waitKey(0)


        
#if __name__ == "__main__":
    #read_flo_file(sys.argv[1])