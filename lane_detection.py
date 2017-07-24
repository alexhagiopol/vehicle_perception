import glob
import os
import pickle
import cv2
import copy
import numpy as np
import shutil
from moviepy.editor import VideoFileClip
from camera_calibration import calibrate
from matplotlib import pyplot as plt


def initialize_lane_detector(pickle_path, images_in_directory=None, videos_in_directory=None, videos_out_directory=None):
    assert (os.path.isfile(pickle_path))
    with open(pickle_path, mode='rb') as f:
        camera_info = pickle.load(f)
    global left_fit
    left_fit = None
    global right_fit
    right_fit = None
    global camera_matrix
    camera_matrix = camera_info['camera_matrix']
    global distortion_coefficients
    distortion_coefficients = camera_info['distortion_coefficients']
    if images_in_directory is not None:  # or args.images_out_directory is not None:
        assert (images_in_directory is not None)
        # assert(args.images_out_directory is not None)
        # assert(os.path.exists(args.images_in_directory))
        procees_still_images(images_in_directory)
    if videos_in_directory is not None or videos_out_directory is not None:
        assert (videos_in_directory is not None)
        assert (videos_out_directory is not None)
        assert (os.path.exists(videos_in_directory))
        process_videos(videos_in_directory, videos_out_directory)


def procees_still_images( images_in_directory):
    global left_fit
    global right_fit
    image_paths = glob.glob(os.path.join(images_in_directory, "*.jpg"))
    for index, image_path in enumerate(image_paths):
        test_image = cv2.imread(image_path)
        left_fit = None
        right_fit = None
        lane_detection_pipeline(test_image, convert_to_RGB=True)


def process_videos(in_video_dir_name, out_video_dir_name):
    global left_fit
    global right_fit
    if os.path.exists(out_video_dir_name):
        shutil.rmtree(out_video_dir_name)
    os.mkdir(out_video_dir_name)
    input_video_filenames = os.listdir(in_video_dir_name)
    for video_filename in input_video_filenames:
        left_fit = None
        right_fit = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        full_video_in_path = os.path.join(os.path.join(dir_path, in_video_dir_name), video_filename)
        assert (os.path.isfile(full_video_in_path))
        full_video_out_path = os.path.join(dir_path, out_video_dir_name)
        print()
        clip = VideoFileClip(os.path.join(in_video_dir_name, full_video_in_path))
        processed_clip = clip.fl_image(lane_detection_pipeline)
        processed_clip_name = os.path.join(full_video_out_path, "processed_" + video_filename)
        processed_clip.write_videofile(processed_clip_name, audio=False)
        print("processed video: ", processed_clip_name)


def lane_detection_pipeline(raw_image, convert_to_RGB=False):
    """
    Execute main processing pipeline.
    """
    global camera_matrix
    global distortion_coefficients
    # convert to RGB
    if convert_to_RGB:
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    # undistort
    undistorted_image = calibrate.undistort_image(raw_image, camera_matrix, distortion_coefficients, visualize=False)
    binary_threshold_image = binary_threshold(undistorted_image, visualize=False)
    homography = compute_forward_to_top_perspective_transform()
    top_view_image = cv2.warpPerspective(binary_threshold_image, homography, (binary_threshold_image.shape[1], binary_threshold_image.shape[0]))
    eroded_top_view = cv2.erode(top_view_image, np.ones((3, 3)))
    top_view_points_left, top_view_points_right, left_radius, right_radius = estimate_lane_lines(eroded_top_view, visualize=False)
    homography_inverse = compute_top_to_forward_perspective_transform()
    front_view_points_left = cv2.perspectiveTransform(top_view_points_left, homography_inverse)
    front_view_points_right = cv2.perspectiveTransform(top_view_points_right, homography_inverse)
    radius = (left_radius + right_radius) / 2
    frame_with_info = display_info(undistorted_image, front_view_points_left, front_view_points_right, radius)
    return frame_with_info


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


def estimate_lane_lines(binary_top_view_image, visualize=False):
    global left_fit
    global right_fit
    if left_fit is None or right_fit is None:  # first lane line estimate
        # estimate search starting point with histogram
        histogram = np.sum(binary_top_view_image[binary_top_view_image.shape[0] // 2:, :], axis=0)
        if visualize:
            out_img = np.dstack((binary_top_view_image, binary_top_view_image, binary_top_view_image)) * 255
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        nwindows = 9  # number of sliding windows
        window_height = np.int(binary_top_view_image.shape[0] / nwindows)
        # x and y positions of all nonzero pixels in the image
        nonzero = binary_top_view_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100  # width of the windows +/- margin
        minpix = 50  # minimum number of pixels found to recenter window
        # empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_top_view_image.shape[0] - (window + 1) * window_height
            win_y_high = binary_top_view_image.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            if visualize:
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

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_top_view_image.shape[0] - 1, binary_top_view_image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    else:  # Lane line estimate already exists
        nonzero = binary_top_view_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_top_view_image.shape[0] - 1, binary_top_view_image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    top_view_points_left = np.float32([[left_fitx[i], ploty[i]] for i in range(len(left_fitx))]).reshape(-1, 1, 2)
    top_view_points_right = np.float32([[right_fitx[i], ploty[i]] for i in range(len(right_fitx))]).reshape(-1, 1, 2)

    # CONVERT FROM PIXELS SPACE TO METERS SPACE
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_meters = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_meters = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curve_rad = ((1 + (2 * left_fit_meters[0] * 719 * ym_per_pix + left_fit_meters[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_meters[0])
    right_curve_rad = ((1 + (
    2 * right_fit_meters[0] * 719 * ym_per_pix + right_fit_meters[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_meters[0])

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
    cv2.fillPoly(image, np.int32([[(polygon_points[i, 0, 0], polygon_points[i, 0, 1]) for i in range(polygon_points.shape[0])]]), [0, 255, 0])
    result = cv2.addWeighted(original_image, 0.5, image, 0.2, 0)
    cv2.putText(result, "lane curvature radius: " + str(int(radius)) + "m", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness=5)
    return result
