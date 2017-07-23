import glob
import os
import argparse
import pickle
import cv2
import copy
import numpy as np
from camera_calibration import calibrate
from matplotlib import pyplot as plt


def compute_forward_to_top_perspective_transform():
    forward_view_points = np.float32([[262, 677], [580, 460], [703, 460], [1040, 677]])
    top_view_points = np.float32([[262, 720], [262, 0], [1040, 0], [1040, 720]])
    forward_view_to_top_view = cv2.getPerspectiveTransform(forward_view_points, top_view_points)
    return forward_view_to_top_view


def compute_top_to_forward_perspective_transform():
    forward_view_points = np.float32([[262, 677], [580, 460], [703, 460], [1040, 677]])
    top_view_points = np.float32([[262, 720], [262, 0], [1040, 0], [1040, 720]])
    top_view_to_forward_view = cv2.getPerspectiveTransform(top_view_points, forward_view_points)
    return top_view_to_forward_view


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


def sliding_window_search(binary_warped, visualize=False):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    # CONVERT TO METERS
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_meters = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_meters = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curve_rad = ((1 + (2 * left_fit_meters[0] * 719 * ym_per_pix + left_fit_meters[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_meters[0])
    right_curve_rad = ((1 + (2 * right_fit_meters[0] * 719 * ym_per_pix + right_fit_meters[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_meters[0])


    '''
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    y_eval = np.max(ploty)
    left_curve_rad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curve_rad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    print("left radius=", left_curve_rad, "right radius=", right_curve_rad)
    '''

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    top_view_points_left = np.float32([[left_fitx[i], ploty[i]] for i in range(len(left_fitx))]).reshape(-1, 1, 2)
    top_view_points_right = np.float32([[right_fitx[i], ploty[i]] for i in range(len(right_fitx))]).reshape(-1, 1, 2)

    if visualize:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        plt.close()

    return top_view_points_left, top_view_points_right, left_curve_rad, right_curve_rad


def display_info(image, left_points, right_points, radius):
    original_image = copy.copy(image)
    polygon_points = np.concatenate((left_points, np.flip(right_points, axis=0)), axis=0)
    # print(np.array([[(polygon_points[i, 0, 0], polygon_points[i, 0, 1]) for i in range(polygon_points.shape[0])]]))
    cv2.fillPoly(image, np.int32([[(polygon_points[i, 0, 0], polygon_points[i, 0, 1]) for i in range(polygon_points.shape[0])]]), [0, 255, 0])
    result = cv2.addWeighted(original_image, 0.5, image, 0.2, 0)
    cv2.putText(result, "lane curvature radius: " + str(int(radius)) + "m", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness=5)
    plt.imshow(result)
    plt.show()
    plt.close()
    return result


def pipeline(raw_image):
    """
    Execute main processing pipeline.
    """
    # convert to RGB
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    # undistort
    undistorted_image = calibrate.undistort_image(raw_image, camera_matrix, distortion_coefficients, visualize=False)
    binary_threshold_image = binary_threshold(undistorted_image, visualize=False)
    homography = compute_forward_to_top_perspective_transform()
    top_view_image = cv2.warpPerspective(binary_threshold_image, homography, (binary_threshold_image.shape[1], binary_threshold_image.shape[0]))
    eroded_top_view = cv2.erode(top_view_image, np.ones((3, 3)))
    top_view_points_left, top_view_points_right, left_radius, right_radius = sliding_window_search(eroded_top_view, visualize=False)
    homography_inverse = compute_top_to_forward_perspective_transform()
    front_view_points_left = cv2.perspectiveTransform(top_view_points_left, homography_inverse)
    front_view_points_right = cv2.perspectiveTransform(top_view_points_right, homography_inverse)
    radius = (left_radius + right_radius) / 2
    # print("left_radius", left_radius, "right_radius", right_radius, "radius", )
    frame_with_info = display_info(undistorted_image, front_view_points_left, front_view_points_right, radius)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate lane line curvature and detect objects in a video file')
    parser.add_argument('--dataset-directory', '-d', dest='dataset_directory', type=str, required=True, help='Required string: Directory containing driving log.')
    parser.add_argument('--pickle-path', '-p', dest='pickle_path', type=str, required=True, help='Required string: Path to pickle file with calibration info.')
    args = parser.parse_args()
    assert(os.path.exists(args.dataset_directory))
    assert(os.path.isfile(args.pickle_path))
    with open(args.pickle_path, mode='rb') as f:
        camera_info = pickle.load(f)
    camera_matrix = camera_info['camera_matrix']
    distortion_coefficients = camera_info['distortion_coefficients']
    image_paths = glob.glob(os.path.join(args.dataset_directory, "*.jpg"))
    for index, image_path in enumerate(image_paths):
        test_image = cv2.imread(image_path)
        pipeline(test_image)
