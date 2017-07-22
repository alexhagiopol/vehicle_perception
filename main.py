import glob
import os
import argparse
import pickle
import cv2
import numpy as np
from camera_calibration import calibrate
from matplotlib import pyplot as plt


def compute_forward_to_top_perspective_transform():
    forward_view_points = np.float32([[262, 677], [580, 460], [703, 460], [1040, 677]])
    top_view_points = np.float32([[262, 720], [262, 0], [1040, 0], [1040, 720]])
    forward_view_to_top_view = cv2.getPerspectiveTransform(forward_view_points, top_view_points)
    return forward_view_to_top_view


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


def show_image(location, title, img, width=3, open_new_window=False):
    if open_new_window:
        plt.figure(figsize=(width, width))
    plt.subplot(*location)
    plt.title(title, fontsize=8)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    if open_new_window:
        plt.show()
        plt.close()


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
        binary_threshold_image = binary_threshold(undistorted_image, visualize=False)
        homography = compute_forward_to_top_perspective_transform()
        top_view_image = cv2.warpPerspective(binary_threshold_image, homography, (binary_threshold_image.shape[1], binary_threshold_image.shape[0]))
        show_image((1, 1, 1), "top_view_image", top_view_image, width=5, open_new_window=True)

