from re import M
import cv2
import numpy as np
from skimage.feature import canny

from utils import _get_frame_patches

MAX_PIXEL_VALUE = 255

def pixel_difference_strategy(input_video: np.array, T1: float, T2: float) -> np.array:
    """
    Calculates the pixel difference between frames to classify shot boundary detection frames.
    A frame is classified as a boundary if the number of pixels that present a difference bigger
    than T1 is greater than T2.
    e.g.: if T1 is 220 and T2 is 0.5, then at least 50% of the frame pixels must present a difference
    greater than 220.
    Parameters:
    input_video: input video as a numpy array with dimensions (NUMBER_OF_FRAMES, FRAME_HEIGHT, FRAME_WIDTH)
    T1         : Threshold related to minimum pixel-wise difference necessary to be classified as boundary.
    T2         : Threshold related to minimum amount of pixels that pass T1 in order to be classified as boundary

    Returns:
    output_video: output video as a numpy array
    """
    # Frame parameters
    FRAME_DIM       = np.prod(input_video.shape[1:]) # frame height * frame width 

    # calculating pixel difference
    pixel_difference           = np.abs(np.diff(input_video, 1, axis = 0))
    percentage_greater_than_T1 = np.count_nonzero(pixel_difference > T1, axis = (1,2)) / FRAME_DIM 
    selected_frames            = ((np.argwhere(percentage_greater_than_T1 > T2)).ravel())
    print('Selected frames: ', selected_frames)

    return selected_frames

def block_difference_strategy(input_video: np.array, T1: float, T2: float) -> np.array:
    """
    Calculates the RMSE between 8x8 patches of adjacent frames to classify as a shot boundary.
    A frame is classified as a boundary if the number of patches that present a RMSE bigger than
    T1 is greater than T2.
    e.g.: if T1 is 10 and T2 is 10, then at least 10 of the frame's patches must present a RMSE greater
    thant 10.

    Parameters:
    input_video: input video as a numpy array with dimension (NUMBER_OF_FRAMES, FRAME_HEIGHT, FRAME_WIDTH)
    T1         : threshold related to the minimum RMSE between frames
    T2         : threshold related to minimum amount of patches that present RMSE greater thant T1

    Returns:
    output_video: output video as a numpy array
    """

    frame_patches              =  _get_frame_patches(input_video)
    patches_rmse               =  np.sqrt((np.diff(np.abs(frame_patches), axis = 0)**2).mean(axis = (3,4)))
    frames_greater_than_T1     =  np.count_nonzero(patches_rmse > T1, axis = (1,2))
    selected_frames            = (np.argwhere(frames_greater_than_T1 >= T2)).ravel()
    
    print('Selected frames: ', selected_frames)
    return selected_frames

def histogram_difference_strategy(input_video: np.array, alpha: float) -> np.array:
    """
    Calculates a simple absolute histogram difference between adjacent frames of input video.
    A given frame is classified as a boundary shot if its absolute histogram difference with the previous
    frame is bigger than the mean + alpha * sigma, where mean and sigma are the mean and the standard
    deviation of the histogram difference series, respectively.

    Parameters:
    input_video: input video as a numpy array with dimension (NUMBER_OF_FRAMES, FRAME_HEIGHT, FRAME_WIDTH)
    alpha      : threshold parameter used to classify frame as a boundary. 
    """

    input_video            = input_video.reshape(input_video.shape[0], -1) # flatten the frames. Resulting shape is (n_frames, frame_height * frame_width)
    histograms             = np.apply_along_axis(lambda x: np.histogram(x, np.arange(256))[0], 1, input_video)
    histograms_differences = np.abs(np.diff(histograms, axis = 0)) # |Hi(j) - H_{i+1}(j)| for all i,j
    histogram_agg_sum      = np.sum(histograms_differences, axis = 1) # SUM(|Hi(j) - H_{i+1}(j)| for all i,j)

    # calculating threshold
    mean  = np.mean(histogram_agg_sum[1:])
    sigma = np.std(histogram_agg_sum[1:])
    threshold = mean + alpha * sigma

    selected_frames = (np.argwhere(histogram_agg_sum > threshold)).ravel()
    print('Selected frames: ', selected_frames)

    return selected_frames
    
def edge_ratio_strategy(input_video: np.array, T1: float) -> np.array:
    """
    Calculates the edge pixel ration between adjacent frames of input video.
    A frame is classified as a edge using the Canny operator with sigma fixed and equal to 2.
    The edge pixel amount ratio is calculated as max(frame_{i + 1}/frame{i}, frame{i}/frame{i+1}). This is used
    since a shot boundary could either increase or decrease drastically the number of edge pixels in a image.

    Parameters:
    input_video: input video as a numpy array with dimension (NUMBER_OF_FRAMES, FRAME_HEIGHT, FRAME_WIDTH)
    T1  : threshold parameter used to classify frame as a boundary. Edge pixel ratio must be greater than T1.
    """
    frame_edges          = np.array(list(map(lambda x: canny(x, sigma = 2), input_video))) #canny operator along frames
    edge_pixel_amount    = np.sum(frame_edges, axis = (1,2)) # sum number of edge pixel in each frame
    
    # Since the first frame is always all 0s, we need to put the edge pixel amount equal to 1
    # to avoid zero-division
    edge_pixel_amount[0] = 1

    # max(frame_{i + 1}/frame{i}, frame{i}/frame{i+1})
    edge_ratio  = np.where(edge_pixel_amount[1:] > edge_pixel_amount[:-1],           
                           np.divide(edge_pixel_amount[1:], edge_pixel_amount[:-1]),
                           np.divide(edge_pixel_amount[:-1], edge_pixel_amount[1:]))

    selected_frames = (np.argwhere(edge_ratio > T1)).ravel()
    print('Selected frames: ', selected_frames)

    return selected_frames


def apply_shot_boundary_detection(input_video, **args):
    print(args)
    detection_method = args.pop('detection_method')
    
    # getting function name and function object for every function defined as _strategy
    techniques = {key[:-9]: value for key,value in globals().items() if key.endswith('_strategy')}
    

    # applying the function technique chosen by user
    if detection_method in techniques.keys():
        return techniques[detection_method](input_video, **args)
    
    # raising error if invalid technique argument
    else:
        message = 'Error in techniques.py - invalid input technique!\n'
        message += 'Available techniques:'
        message += ''.join([f'\n\t- {technique}' for technique in techniques.keys()])
        raise NotImplementedError(message)