import os
import subprocess
import inspect

imgDir = "../ridges/"
lst = os.listdir(imgDir)
lst_aux = lst[:]
for filename in lst_aux:   
    if 'Identifier' in filename:
        lst.remove(filename)
        print('ciao', filename)



#print('lst', lst)
number_files = len(lst) 
print('\n number file',number_files)
#print('\n half number file rounded', int(number_files/2))


script_directory = os.path.dirname(os.path.abspath(
inspect.getfile(inspect.currentframe())))
 
print('script directory' , script_directory)

for i in range(int(number_files/2)):
    root_filename = os.path.join(script_directory, imgDir, 'frame'+str(i))
    img1 = os.path.join(root_filename+'_0.jpg')
    img2 = os.path.join(root_filename+'_1.jpg')
    frame_match_name = 'frame_match{}'.format(i)
#print('\nroot_filename', root_filename)
#print('\nimg1', img1)
#print('\nimg2', img2)
#print('\frame_match_name', frame_match_name)

    #process = subprocess.call(['cd', 'deepmatching_1.2.2_c++/'])
    process = subprocess.run(['./deepmatching', img1, img2, '-jpg_settings', '-out' , frame_match_name])
    # print(process)
    # stdout, stderr = process.communicate()