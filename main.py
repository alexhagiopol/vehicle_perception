import glob
import os
import argparse
import pickle
import cv2
import numpy as np
from camera_calibration import calibrate
from matplotlib import pyplot as plt


def binary_threshold(undistorted_image, visualize=False):
    # Convert to HLS color space and separate the S channel
    # Note: undistorted_image is the undistorted image
    hls = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 255) | (sxbinary == 255)] = 1

    if visualize:
        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')
        plt.show()
        plt.close()
    return combined_binary


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
        undistorted_image = calibrate.undistort_image(test_image, camera_matrix, distortion_coefficients, visualize=False)
        binary_threshold_image = binary_threshold(undistorted_image, visualize=True)
