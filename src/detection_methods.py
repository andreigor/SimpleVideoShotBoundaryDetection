from select import select
import cv2
import numpy as np

MAX_PIXEL_VALUE = 255

def pixel_difference_strategy(input_video: np.array, T1: float, T2: float) -> np.array:
    """
    Calculates the pixel difference between frames to classify shot boundary detection frames.
    A frame is classified as a boundary if the number of pixels that present a difference bigger
    than T1 is greater than T2.
    e.g.: if T1 is 0.8 and T2 is 0.5, then at least 50% of the frame pixels must present a difference
    greater than 80% of the frame max pixel value.

    Parameters:
    input_video: input video as a numpy array 
                 with dimensions (NUMBER_OF_FRAMES, FRAME_HEIGHT, FRAME_WIDTH)

    T1         : Threshold related to minimum pixel-wise difference necessary 
                 to be classified as boundary.

    T2         : Threshold related to minimum amount of pixels that pass T1 
                 in order to be classified as boundary

    Returns:
    output_video: output video as a numpy array
    """
    # Frame parameters
    MAX_PIXEL_VALUE = np.max(input_video)
    FRAME_DIM       = np.prod(input_video.shape[1:]) # frame height * frame width 
    

    pixel_difference           = np.abs(np.diff(input_video, 1, axis = 0))
    percentage_greater_than_T1 = np.count_nonzero(pixel_difference > T1*MAX_PIXEL_VALUE, axis = (1,2))
    selected_frames            = ((np.argwhere(percentage_greater_than_T1 > T2 * FRAME_DIM)).ravel())

    return selected_frames



def apply_chosen_technique(input_video, chosen_technique, *args):
    # getting function name and function object for every function defined as _strategy
    techniques = {key[:-9]: value for key,value in globals().items() if key.endswith('_strategy')}
    

    # applying the function technique chosen by user
    if chosen_technique in techniques.keys():
        return techniques[chosen_technique](input_video, *args)
    
    # raising error if invalid technique argument
    else:
        message = 'Error in techniques.py - invalid input technique!\n'
        message += 'Available techniques:'
        message += ''.join([f'\n\t- {technique}' for technique in techniques.keys()])
        raise NotImplementedError(message)

