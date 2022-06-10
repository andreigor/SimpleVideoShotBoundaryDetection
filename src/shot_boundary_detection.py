import sys
import argparse



from utils import *
from detection_methods import apply_shot_boundary_detection

def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_video', help = 'input mp4 ou mpg video', type = str)
    parser.add_argument('output_video', help = 'input mp4 ou mpg video', type = str)

    subparser            = parser.add_subparsers(dest = 'detection_method', help = 'shot boundary detection method: pixel_difference, block_difference, histogram_difference, edge_ratio', required=True)

    pixel_difference     = subparser.add_parser('pixel_difference')
    block_difference     = subparser.add_parser('block_difference')
    histogram_difference = subparser.add_parser('histogram_difference')
    edge_ratio           = subparser.add_parser('edge_ratio')

    pixel_difference.add_argument('--T1', type = float, required = True, help = 'minimum pixel difference')
    pixel_difference.add_argument('--T2', type = float, required = True, help = 'minimum number of pixels greater than T1')

    block_difference.add_argument('--T1', type = float, required = True, help = 'minimum block difference')
    block_difference.add_argument('--T2', type = float, required = True, help = 'minimum number of blocks greater than T1')

    histogram_difference.add_argument('--alpha', type = float, required = True, help = 'standart deviation threshold constant')

    edge_ratio.add_argument('--T1', type = float, required = True, help = 'minimum edge ratio threshold')

    args = vars(parser.parse_args())
    return args

def main():
    # reading input parameters
    args = parse_input_args()
    input_video_path  = args.pop('input_video')
    output_video_path = args.pop('output_video')
    
    # reading input video
    input_video           = read_video_as_numpy_hsv_array(input_video_path)
    grayscale_input_video = input_video[:,:,:,2].copy()

    # the first video frame is just a all-zeros matrix, used to aid shot boundary detection calculation.
    # it is important to remove it from the input video
    input_video           = np.delete(input_video, 0, axis = 0) 

    # applying shot boundary detection
    selected_frames = apply_shot_boundary_detection(grayscale_input_video, **args)

    # save and show output video
    save_output_video(input_video, selected_frames, output_video_path)
    output_video = input_video[selected_frames]


    # PART JUST FOR DEBUG
    for frame in output_video:
        cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_HSV2BGR))


        # Press Q on keyboard to  exit
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    