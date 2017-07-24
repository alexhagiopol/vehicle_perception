Get the project:
    
    git clone git@github.com:alexhagiopol/vehicle_perception.git
    cd vehicle_perception
    git submodule update --init

Get example datasets:

    wget -O datasets.zip "https://www.dropbox.com/s/uzual4vchhzms1d/datasets.zip?dl=1"
    unzip datasets.zip

Perform camera calibration:

    python camera_calibration/calibrate.py -ch 6 -cw 9 -cd datasets/calibration_images/ -p camera_info.p

Run the pipeline:
    
    python main.py -p camera_info.p -vid datasets/test_videos/ -vod output/
    
TODO List:
* ~~Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.~~
* ~~Apply a distortion correction to raw images.~~
* ~~Use color transforms, gradients, etc., to create a thresholded binary image.~~
* ~~Apply a perspective transform to rectify binary image ("birds-eye view").~~
* ~~Detect lane pixels and fit to find the lane boundary.~~
* ~~Determine the curvature of the lane and vehicle position with respect to center.~~
* ~~Warp the detected lane boundaries back onto the original image.~~
* ~~Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.~~

[![fpv](figures/video_preview.png)](https://youtu.be/l6JVoMymH2s#t=0s "Lane Detection Demo")

