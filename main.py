import argparse
import lane_detection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate lane line curvature and detect objects in a video file')
    parser.add_argument('--pickle-path', '-p', dest='pickle_path', type=str, required=True,
                        help='Required string: Path to pickle file with calibration info.')
    parser.add_argument('--images-in-directory', '-iid', dest='images_in_directory', type=str, required=False,
                        help='Required string: Input directory containing images.')
    parser.add_argument('--images-out-directory', '-iod', dest='images_out_directory', type=str, required=False,
                        help='Required string: Output directory containing images.')
    parser.add_argument('--videos-in-directory', '-vid', dest='videos_in_directory', type=str, required=False,
                        help='Required string: Input directory containing videos.')
    parser.add_argument('--videos-out-directory', '-vod', dest='videos_out_directory', type=str, required=False,
                        help='Required string: Output directory containing videos.')
    args = parser.parse_args()
    lane_detection.initialize_lane_detector(args.pickle_path, args.images_in_directory, args.videos_in_directory, args.videos_out_directory)
