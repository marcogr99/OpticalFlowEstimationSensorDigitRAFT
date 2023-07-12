import os
import subprocess
import inspect

imgDir = "../ridges/"
lst_frame = os.listdir(imgDir)
lst_aux=lst_frame[:]
lst0=[]
lst1=[]
lst_out = []
lst_edges=[]
lst_match=[]

for filename in lst_aux:   
     if 'Identifier' in filename:
         lst_frame.remove(filename)
        # print('ciao', filename)

# for images in lst_frame:
#     if '_0.jpg' in images:
#         img= imgDir + images 
#         lst0.append(img)
#     elif '_1.jpg' in images:
#         img= imgDir + images 
#         lst1.append(img)

# lst0=sorted(lst0,key=len)
# lst1=sorted(lst1,key=len)


number_files = len(lst_frame)/2

# edgeDir = "frame_edges/"
# lst_edges = os.listdir(edgeDir)
# print('\n lst_edges:',lst_edges)



# matchDir = "frame_match/"
# lst_match = os.listdir(matchDir)
# print('\n lst_match:',lst_match)



for i in range(int(number_files)):
    root_filename = os.path.join( imgDir, 'frame'+str(i))
    root_edge= os.path.join( '../frame_edges', 'frame'+str(i))
    img0 = os.path.join(root_filename+'_0.jpg')
    img1 = os.path.join(root_filename+'_1.jpg')
    edge= os.path.join(root_edge +'_edge')
    match=os.path.join('frame_match','frame_match'+str(i))
    frame_flow_name = 'frame_flow{}.flo'.format(i)
    
    lst1.append(img1)
    lst0.append(img0)
    lst_edges.append(edge)
    lst_match.append(match)
    lst_out.append(frame_flow_name)

#print('\n lst_match:',lst_match)
#print('\n lst_edges:',lst_edges)

    
#print('lista out:', lst_out)
for img1,img2,edge,match,out in zip(lst0, lst1, lst_edges,lst_match,lst_out):
    process = subprocess.run(['./epicflow', img1, img2, edge, match , out])

print('\n img1:',img1)
print('\n img2:',img2)
print('\n edge:',edge)
print('\n match:',match)
print('\n out:',out)