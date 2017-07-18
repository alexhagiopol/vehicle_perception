import glob
import os
import argparse
import pickle
import cv2
from camera_calibration import calibrate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate lane line curvature and detect objects in a video file')
    parser.add_argument('--dataset-directory', '-d', dest='dataset_directory', type=str, required=True, help='Required string: Directory containing driving log.')
    parser.add_argument('--pickle-path', '-p', dest='pickle_path', type=str, required=True, help='Required string: Path to pickle file with calibration info.')
    args = parser.parse_args()
    with open(args.pickle_path, mode='rb') as f:
        camera_info = pickle.load(f)
    camera_matrix = camera_info['camera_matrix']
    distortion_coefficients = camera_info['distortion_coefficients']
    image_paths = glob.glob(os.path.join(args.dataset_directory, "*.jpg"))
    for index, image_path in enumerate(image_paths):
        test_image = cv2.imread(image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        undistorted_image = calibrate.undistort_image(test_image, camera_matrix, distortion_coefficients, visualize=True)
