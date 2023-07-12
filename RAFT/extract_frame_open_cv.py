from datetime import timedelta
import cv2
import numpy as np
import os

# i.e if video of duration 30 seconds, saves 10 frame per second = 300 frames saved in total
SAVING_FRAMES_PER_SECOND = 4
def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def main(video_file, output_dir, distincFrame):
    filename, _ = os.path.splitext(video_file)
    filename += "-opencv"
    # make a folder by the name of the video file if output_dir not specified
    if output_dir == None:
        if not os.path.isdir(filename):
            os.mkdir(filename)
    else:
        output_dir, _ = os.path.splitext(output_dir)
        #output_dir += "-opencv" 
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    # read the video file    
    cap = cv2.VideoCapture(video_file)
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # start the loop
    aux = 1
    count = 1
    #distincFrame = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            if aux%2 == 0:
                a=1
            else:
                a=0
            filen = "frame"+str(distincFrame)+"_"+str(a)+".jpg"
            print(filen)
            if (aux%2 == 0) :
                distincFrame += 1
            
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, filen), frame) 
                #print('\nsalvo i file in', os.path.join(output_dir, filen))
            else:
                cv2.imwrite(os.path.join(filename, filen), frame)
                #print('\nsalvo i file in', os.path.join(filename, filen))

            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
                aux += 1
            except IndexError:
                pass
        # increment the frame count
        count += 1
    return distincFrame

if __name__ == "__main__":
    import sys
    import glob
    from pathlib import Path
    distincFrame=0
    p = Path(str(sys.argv[1])+'/')
    print('path input',p)
    for folder in p.iterdir(): 
        # All subdirectories in the current directory, not recursive.
        print('\ndirectory in cui entro', folder, '\n')
        if folder.is_dir():
            for video in glob.iglob(os.path.join(folder, '*.mp4')): # prendo tutti i file .mp4 nella directory indicata come primo elemento nella linea di comando
                
                if len(sys.argv)<3:
                    output_dir = None
                else:
                    output_dir = sys.argv[2]
                video_file=video
                distincFrame = main(video_file, output_dir, distincFrame) #mi faccio restitutire il contatore di frame così da avere un'unica sequenza di frame e riparto
                #main(video_file)
