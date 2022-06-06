import numpy as np
import cv2


def read_video_as_numpy_array(video_path):
    cap = cv2.VideoCapture(video_path)
    frameWidth, frameHeight = _get_frame_config(cap) 
    
    grayscale_frames = [np.zeros(shape = (frameHeight, frameWidth))]
    normal_frames    = [np.zeros(shape = (frameHeight, frameWidth, 3))]

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read() # first frame
        if ret == True:
            normal_frames.append(frame)

            # transform to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayscale_frames.append(frame)
        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    #
    numpy_grayscale_video = np.array(grayscale_frames, dtype = np.dtype('uint8'))
    numpy_normal_video    = np.array(normal_frames, dtype = np.dtype('uint8'))

    return numpy_normal_video, numpy_grayscale_video


def _get_frame_config(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
