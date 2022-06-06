import sys
import argparse
from tracemalloc import start



from utils import *
from detection_methods import apply_chosen_technique

class InputParameterError(Exception):
    "Raised when input parameter is not as expected"
    pass

def check_input_parameters():
    if len(sys.argv) < 5:
        message = 'Error in shot_boundary_detection.py:\n '
        message += '<P1> <P2> <P3> <P4>\n'
        message += 'P1: Input video\nP2: Detection method.\nP3: Outuput video.\nP4+: Method depedent parameters\n'

        raise InputParameterError(message)
    
def main():
    # input parameters
    check_input_parameters()
    parameters = list(map(lambda x: float(x), sys.argv[4:]))


    input_video, grayscale_input_video = read_video_as_numpy_array(sys.argv[1])
    selected_frames                    = apply_chosen_technique(grayscale_input_video ,sys.argv[2], *parameters)
    save_output_video(input_video, selected_frames, sys.argv[3])
    output_video = input_video[selected_frames]

    for frame in output_video:
        cv2.imshow('Frame', frame)


        # Press Q on keyboard to  exit
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    