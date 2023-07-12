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

def main():
    print('len sys argv', len(sys.argv))
    print('sys argv 0',sys.argv[0])
    print('sys argv 1',sys.argv[1])
    print('sys argv 2',sys.argv[2])
    print('sys argv 3',sys.argv[3])
    print('sys argv 4',sys.argv[4])
    if len(sys.argv) == 5: 
      image_idxs = sys.argv[2]
      iterations = sys.argv[3]
    
    elif len(sys.argv) == 6:
      image_idxs = sys.argv[3]
      iterations = sys.argv[4]
    flow = horn_schunck(image_idxs, iterations)


def horn_schunck(image_idxs, iterations):
  flow_dm     = np.zeros((1,3))
  alpha       = 0.1
  sub_img_dim = 300
  idx         = 0
  iterations = int(iterations)
  
  p = Path(str(sys.argv[1])+'/')

  # All subdirectories in the current directory, not recursive.
  for f in p.iterdir(): 
    
    if f.is_dir():
      image_dir = f
      lst = os.listdir(image_dir) # your directory path
      number_files = len(lst) 
      print('\n number file',number_files)
      print('\n half number file rounded', int(number_files/2))
      print('\nfolder', f, '\n')

    for i in range(int(number_files/2)):
      root_filename = os.path.join(image_dir, 'frame'+str(i))
      img1 = os.path.join(root_filename+'_0.jpg')
      img2 = os.path.join(root_filename+'_1.jpg')    
       
      
      #img1_path=str(sys.argv[1])
      img1_path=img1
      image_1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
      im1 = cv.imread(img1_path, cv.IMREAD_COLOR)
      #cv.imshow('img1 Gray scale', image_1)
      #cv.imshow('img1', im1)

      #img2_path=str(sys.argv[2])
      img2_path=img2
      image_2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
      im2 = cv.imread(img2_path, cv.IMREAD_COLOR)
      #cv.imshow('img2 Gray scale', image_2)
      #cv.imshow('img2', im2)

      frame_start = image_1
      u      = np.zeros_like(frame_start, dtype=float)
      v      = np.zeros_like(frame_start, dtype=float)
      dudx   = np.zeros_like(frame_start, dtype=float)
      dvdx   = np.zeros_like(frame_start, dtype=float)
      dudy   = np.zeros_like(frame_start, dtype=float)
      dvdy   = np.zeros_like(frame_start, dtype=float)
      div    = np.zeros_like(frame_start, dtype=float)
      u_mean = np.zeros_like(frame_start, dtype=float)
      v_mean = np.zeros_like(frame_start, dtype=float)
      Ex     = np.zeros_like(frame_start, dtype=float)
      Ey     = np.zeros_like(frame_start, dtype=float)
      Et     = np.zeros_like(frame_start, dtype=float)

      frame_idx_prev = image_idxs[0]
      exec_time      = 0

      # repeat till end frame was reached
      for frame_idx in image_idxs[1:]:
        start_time = time.time()
        idx        = idx+1
        #print('frame_idx', frame_idx)
        #print('frame index prev', frame_idx_prev)

        if frame_idx_prev == frame_idx:
          flow_dr = np.array([[idx,0.0,exec_time]])
        else:
          #frame_end  = images[frame_idx-1][95:(95+sub_img_dim), 150:(150+sub_img_dim)]
          frame_end = image_2
          #print('frame_end',frame_end)

          #ret, frame_end = cv.threshold(frame_end, 50, 255, cv.THRESH_BINARY_INV)

          #blackout area outside circle to select the region of the tactip with pins only
          #frame_end  = images[frame_idx-1][30:(30+sub_img_dim), 100:(100+sub_img_dim)]
          #mask = np.zeros_like(frame_end)
          #mask = cv.circle(mask, (205,207), 195, (255,255,255), -1)
          #frame_end = cv.bitwise_and(frame_end, mask) 
          #ret, frame_end = cv.threshold(frame_end, 250, 255, cv.THRESH_BINARY_INV)

          rows_num   = frame_start.shape[0]
          cols_num   = frame_start.shape[1]
          pixels_num = rows_num*cols_num
  
          u = np.zeros_like(frame_start, dtype=float)
          v = np.zeros_like(frame_start, dtype=float)
      
          frame_start_f = frame_start.astype(dtype=float)
          frame_end_f   = frame_end.astype(dtype=float)
          #print('frame start and end',frame_start, frame_end)

          frame_start    = frame_end
          frame_idx_prev = frame_idx
          #print('frame start f and end f',frame_start_f, frame_end_f)

          Ex = (1.0/4.0)*(np.pad(frame_start_f[:,1:] ,((0,0),(0,1)),'edge') - frame_start_f +
                          np.pad(frame_start_f[1:,1:],((0,1),(0,1)),'edge') - np.pad(frame_start_f[1:,:],((0,1),(0,0)),'edge') +
                          np.pad(frame_end_f[:,1:]   ,((0,0),(0,1)),'edge') - frame_end_f +
                          np.pad(frame_end_f[1:,1:]  ,((0,1),(0,1)),'edge') - np.pad(frame_end_f[1:,:]  ,((0,1),(0,0)),'edge'))

          Ey = (1.0/4.0)*(np.pad(frame_start_f[1:,:] ,((0,1),(0,0)),'edge') - frame_start_f +
                          np.pad(frame_start_f[1:,1:],((0,1),(0,1)),'edge') - np.pad(frame_start_f[:,1:],((0,0),(0,1)),'edge') +
                          np.pad(frame_end_f[1:,:]   ,((0,1),(0,0)),'edge') - frame_end_f +
                          np.pad(frame_end_f[1:,1:]  ,((0,1),(0,1)),'edge') - np.pad(frame_end_f[:,1:]  ,((0,0),(0,1)),'edge'))

          Et = (1.0/4.0)*(frame_end_f                                     - frame_start_f +
                          np.pad(frame_end_f[1:,:] ,((0,1),(0,0)),'edge') - np.pad(frame_start_f[1:,:] ,((0,1),(0,0)),'edge') +
                          np.pad(frame_end_f[:,1:] ,((0,0),(0,1)),'edge') - np.pad(frame_start_f[:,1:] ,((0,0),(0,1)),'edge') +
                          np.pad(frame_end_f[1:,1:],((0,1),(0,1)),'edge') - np.pad(frame_start_f[1:,1:],((0,1),(0,1)),'edge'))
          #print('EX, EY, ET\n', Ex, Ey, Et)
          for rep_num in range(iterations):
            u_mean = (1.0/6.0)*(np.pad(u[:-1,:],((1,0),(0,0))) +
                                np.pad(u[:,1:] ,((0,0),(0,1))) +
                                np.pad(u[1:,:] ,((0,1),(0,0))) +
                                np.pad(u[:,:-1],((0,0),(1,0)))) + \
                      (1.0/12.0)*(np.pad(u[:-1,:-1],((1,0),(1,0))) +
                                np.pad(u[1:,1:]    ,((0,1),(0,1))) +
                                np.pad(u[:-1,1:]   ,((1,0),(0,1))) +
                                np.pad(u[1:,:-1]   ,((0,1),(1,0))))

            v_mean = (1.0/6.0)*(np.pad(v[:-1,:],((1,0),(0,0))) +
                                np.pad(v[:,1:] ,((0,0),(0,1))) +
                                np.pad(v[1:,:] ,((0,1),(0,0))) +
                                np.pad(v[:,:-1],((0,0),(1,0)))) + \
                      (1.0/12.0)*(np.pad(v[:-1,:-1],((1,0),(1,0))) + 
                                np.pad(v[1:,1:]    ,((0,1),(0,1))) +
                                np.pad(v[:-1,1:]   ,((1,0),(0,1))) +
                                np.pad(v[1:,:-1]   ,((0,1),(1,0))))

            u = u_mean - Ex*(Ex*u_mean + Ey*v_mean + Et)/(alpha**2 + Ex**2 + Ey**2)   
            v = v_mean - Ey*(Ex*u_mean + Ey*v_mean + Et)/(alpha**2 + Ex**2 + Ey**2)
      
          dudx = np.pad(u[:,1:],((0,0),(0,1))) - np.pad(u[:,:-1],((0,0),(1,0)))
          # dvdx = np.pad(v[:,1:],((0,0),(0,1))) - np.pad(v[:,:-1],((0,0),(1,0)))
          # dudy = np.pad(u[1:,:],((0,1),(0,0))) - np.pad(u[:-1,:],((1,0),(0,0)))
          dvdy = np.pad(v[1:,:],((0,1),(0,0))) - np.pad(v[:-1,:],((1,0),(0,0)))
          div  = dudx + dvdy
          #print('\ndudx dvdx\n',dudx,dvdy)

          flow_dr = np.hstack((np.ones((pixels_num,1))*idx,
                               div.reshape(pixels_num,1),
                               np.ones((pixels_num,1))*exec_time))
          #print('div', div)
          #print('flow dr', flow_dr)

        flow_dm   = np.append(flow_dm, flow_dr, axis=0)
        exec_time = time.time() - start_time
      flow_df = pd.DataFrame(flow_dm, columns = ['idx','div','exec_time'])

      flow = np.array([u,v])
      np.save('Flow_gt.npy', flow)
      flow_uv = flow.transpose(1,2,0)

      flow_img = flow2rgb(flow , None)
      flow_img=2 * (flow_img / 255.0) - 1.0
      flow_img1 = (flow_img*255).astype(np.uint8).transpose(1,2,0)
      
      cv.imshow('flow image',flow_img1)
      cv.waitKey(0)

      '''
      import flow_vis
      flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
      
      #flow_color = cv.medianBlur(flow_color, 3)
      #cv.imshow('flow image', (flow_img*255).astype(np.uint8).transpose(1,2,0))
      #cv.imshow('flow image 2',flow_color)

      # convert to grayscale
      gray = cv.cvtColor(flow_color, cv.COLOR_BGR2GRAY)

      # blur
      blur = cv.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)

      # divide
      divide = cv.divide(gray, blur, scale=255)

      # otsu threshold
      thresh = cv.threshold(divide, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

      # apply morphology
      kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
      morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

      # display it
      #cv.imshow("gray", gray)
      #cv.imshow("divide", divide)
      #cv.imshow("thresh", thresh)
      #cv.imshow("morph", morph)
      #cv.waitKey(0)
      

  #--------------------------------------------------------------------------------------
  #PROVO A MIGLIORARE L'IMMAGINE
  #--------------------------------------------------------------------------------------
      # PRIMO METODO
      #image = flow_color  # Load the input image
      flow_img = (flow_img*255).astype(np.uint8).transpose(1,2,0)
      image = flow_img
      blurred = cv.GaussianBlur(image, (0, 0), 3)  # Apply Gaussian blur
      sharpened = cv.addWeighted(image, 1.5, blurred, -0.5, 0)  # Apply unsharp masking
      #cv.imshow('output_image.jpg 1', sharpened)  # Save the sharpened image

      # SECONDO METODO
      image = flow_color  # Load the input image
      image = flow_img
      gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert to grayscale
      laplacian = cv.Laplacian(gray, cv.CV_8U)  # Apply Laplacian filter
      sharpened = cv.add(image, np.dstack([laplacian] * 3))  # Add the Laplacian image to the original
      #cv.imshow('output_image.jpg 2', sharpened)  # Save the sharpened image

      # TERZO METODO
      #image = flow_color  # Load the input image
      image = flow_img
      lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)  # Convert to LAB color space
      l, a, b = cv.split(lab)  # Split the LAB channels
      clahe = cv.createCLAHE(clipLimit=2.0)  # Create a CLAHE object
      l = clahe.apply(l)  # Apply CLAHE on the L channel
      lab = cv.merge([l, a, b])  # Merge the LAB channels
      enhanced = cv.cvtColor(lab, cv.COLOR_LAB2BGR)  # Convert back to BGR color space
      #cv.imshow('output_image.jpg 3', enhanced)  # Save the contrast-enhanced image

      # PROVO A METTERE INSIEME I METODI SOPRA
      #image__ = flow_color
      image__ = flow_img
      blurred1 = cv.GaussianBlur(image__, (0, 0), 3)  # Apply Gaussian blur
      sharpened1 = cv.addWeighted(image__, 1.5, blurred1, -0.5, 0)  # Apply unsharp masking
      gray1 = cv.cvtColor(sharpened1, cv.COLOR_BGR2GRAY)  # Convert to grayscale
      laplacian1 = cv.Laplacian(gray1, cv.CV_8U)  # Apply Laplacian filter
      sharpened2 = cv.add(sharpened1, np.dstack([laplacian] * 3))  # Add the Laplacian image to the original
      lab = cv.cvtColor(sharpened2, cv.COLOR_BGR2LAB)  # Convert to LAB color space
      l, a, b = cv.split(lab)  # Split the LAB channels
      clahe = cv.createCLAHE(clipLimit=2.0)  # Create a CLAHE object
      l = clahe.apply(l)  # Apply CLAHE on the L channel
      lab = cv.merge([l, a, b])  # Merge the LAB channels
      enhanced2 = cv.cvtColor(lab, cv.COLOR_LAB2BGR)  # Convert back to BGR color space

      #cv.imshow('output tot', enhanced2)

      denoised = cv.fastNlMeansDenoisingColored(sharpened, None, 3, 3, 7, 15)  # Apply non-local means denoising
      #cv.imshow('denoised_image.jpg', denoised)
      #cv.waitKey(0)
      '''
      
      if len(sys.argv)==5:
        output_folder_path = Path(str(sys.argv[4]))
      elif len(sys.argv)==6:
        output_folder_path = Path(str(sys.argv[5]))
      #output_dir = output_folder_path.parent.parent.name + '/' + 'flow_digit'
      output_dir = os.path.join(output_folder_path.name, 'flow_digit')
      if not os.path.isdir(output_dir):
         os.mkdir(output_dir)
      filen = os.path.basename(img1_path)
      print('\n Salvo il file in:', os.path.join(output_dir, filen))
      cv.imwrite(os.path.join(output_dir, filen), flow_img)


  return flow_df, flow_img

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


if __name__ == "__main__":
  # QUANDO LANCI IL FILE DA TERMINALE INDICARE:
  #      - PRIMO ARGOMENTO: PATH TO FRAME 1
  #      - SECONDO ARGOMENTO: PATH TO FRAME 2
  #      - TERZO ARGOMENTO : IMAGE_IDXS
  #      - QUARTO ARGOMENTO : ITERATIONS
  #      - QUINTO ARGOMENTO : DIRECTORY DOVE VOGLIO SALVARE LE IMMAGINI DI FLUSSO
  #  OPPURE
  #      - PRIMO ARGOMENTO: PATH TO DIRECTORY DOVE SONO I FRAME
  #      - SECONDO ARGOMENTO: IMAGE_IDXS
  #      - TERZO ARGOMENTO : ITERATIONS
  #      - QUARTO ARGOMENTO : DIRECTORY DOVE VOGLIO SALVARE LE IMMAGINI DI FLUSSO


  main()
