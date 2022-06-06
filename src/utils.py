import numpy as np
from skimage.util import view_as_blocks
import cv2


def read_video_as_numpy_array(video_path):
    cap = cv2.VideoCapture(video_path)
    frameWidth, frameHeight = _get_frame_config(cap) 
    
    grayscale_frames = [np.zeros(shape = (frameHeight, frameWidth))]
    normal_frames    = []

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

def save_output_video(input_video, selected_frames, video_name):
    """
    Save output video in mp4 format.
    """
    size = (input_video.shape[0], input_video.shape[1])
    fps = 2
    video = cv2.VideoWriter(video_name + '.mp4', cv2.VideoWriter_fourcc(*'X264'), fps, (size[0], size[1]), False)

    for frame in selected_frames:
        video.write(input_video[frame])
    video.release()


def _get_frame_config(cap):
    """
    Gets frame width and frame height.
    """
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def _get_frame_patches(numpy_video, patch_size = (8,8)):
    """
    Extracts (n x n) frame patches from a given video represented as numpy array.
    """
    return view_as_blocks(numpy_video, (1, *patch_size)).squeeze()