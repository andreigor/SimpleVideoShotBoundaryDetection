import sys



from utils import *
from detection_methods import apply_chosen_technique

class InputParameterError(Exception):
    "Raised when input parameter is not as expected"
    pass


def main():
    # input parameters
    if len(sys.argv) != 4:
        message = 'Error in shot_boundary_detection.py:\n '
        message += '<P1> <P2> <P3>\n'
        message += 'P1: Input video\nP2: Detection method.\nP3: Outuput video. \n'
        raise InputParameterError(message)

    input_video, grayscale_input_video = read_video_as_numpy_array(sys.argv[1])
    boundary_frames = apply_chosen_technique(grayscale_input_video ,sys.argv[2], 0.8, 0.5)
    print(boundary_frames)
    output_video = input_video[boundary_frames]

    for frame in output_video:
        cv2.imshow('Frame', frame)


        # Press Q on keyboard to  exit
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
    