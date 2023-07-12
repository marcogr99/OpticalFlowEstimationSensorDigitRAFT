import numpy as np
import os
import sys

# def read_flo_filee(filename):
#     with open(filename, 'rb') as f:
#         magic = np.fromfile(f, np.float32, count=1)
#         width = np.fromfile(f, np.int32, count=1)[0]
#         height = np.fromfile(f, np.int32, count=1)[0]
#         data = np.fromfile(f, np.float32, count=2*width*height)
#         flow = np.resize(data, (height, width, 2))

#         print('\n width:',width)
#         print('\n height:',height)
#         print('\n shape:',np.fromfile(f, np.int32, count=1))
#     return flow

# def flow_to_uv__(flow):
#     u = flow[:, :, 0]
#     v = flow[:, :, 1]
#     return u, v

# # Esempio di utilizzo
# flow = read_flo_filee('frame_flow0.flo')
# u, v = flow_to_uv__(flow)
# print('\n u:',u)
# print('\n v:',v)
# np.save('Flow_gt0.npy',np.array([u,v]))

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            #print('\ndata, h, w(frame_utils):',data, h, w)
            flow=np.resize(data, (int(h), int(w), 2))
            return flow

def flow_to_uv__(flow):
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    return u, v

flowDir = "frame_flo/"
print(flowDir)
lst_frame = os.listdir(flowDir)

number_files = len(lst_frame)
for i in range (int(number_files)):
    
    flow = readFlow('frame_flo/frame_flow{}.flo'.format(i))
    u, v = flow_to_uv__(flow)
    
    print('\n u:',u)
    print('\n v:',v)
    np.save('matrici_u_v_gt/Flow_gt{}.npy'.format(i),np.array([u,v]))