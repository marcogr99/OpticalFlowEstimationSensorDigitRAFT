import os
import subprocess
import inspect


flowDir = "../EpicFlow_v1.00/frame_flo/"
lst_frame = os.listdir(flowDir)

number_files = len(lst_frame)

for i in range(int(number_files)):
    
    flo_file=os.path.join(flowDir,'frame_flow{}.flo'.format(i))
    out_name_flo_png = 'frame{}_0.png'.format(i)


    process = subprocess.run(['./color_flow', flo_file, out_name_flo_png])
    print('\n visualizzazione frame:',i)