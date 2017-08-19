# Advanced Lane Finding

--------------------------------------------------------------------------------

## TL;DR

Driving lane tracking on video taken from dash camera

## Table of Contents 
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Advanced Lane Finding](#advanced-lane-finding)
	- [TL;DR](#tldr)
	- [Table of Contents](#table-of-contents)
	- [Project Structure](#project-structure)
	- [Camera Calibration](#camera-calibration)
	- [Pipeline (single images)](#pipeline-single-images)
		- [1\. Distortion correction.](#1-distortion-correction)
		- [2\. Color and gradient thresholding](#2-color-and-gradient-thresholding)
		- [3\. Perspective transform](#3-perspective-transform)
		- [4\. Lane line fitting](#4-lane-line-fitting)
			- [5\. Lane curvature and offset calculation](#5-lane-curvature-and-offset-calculation)
			- [6\. Lane marking](#6-lane-marking)
	- [Pipeline (video)](#pipeline-video)
	- [Discussion](#discussion)

<!-- /TOC -->

--------------------------------------------------------------------------------

## Project Structure

My project includes the following files:

- lane_tracking.py: containing the core lane tracking functions and script to run the lane tracking pipeline on a video
- pipeline.ipynb: demonstrate the intermediate results of the pipeline
- video_output.mp4 is the video output of running `lane_tracking.py` on `project_video.mp4`
- README.md summarizing the results

## Camera Calibration

The code for this step is contained in function `camera_calibration` in `lane_tracking.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. It's worth noting that the number of corners is 9x5 for the first calibration image, but 9x6 for the rest.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

## Pipeline (single images)

### 1\. Distortion correction.

The first step of the pipeline is apply distortion correction on each of the images, achieved by calling `cv2.undistort()` and the distortion matrices calculated by the camera calibration step.

### 2\. Color and gradient thresholding

I used a combination of color and gradient thresholds to generate a binary image. See function `color_gradient_thresh(img)` starting from line #91 in `lane_tracking.py`

```python
def color_gradient_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2]
    sobelx_mask = sobelx_thresh(gray, sobel_kernel=7, thresh=(25, 120))
    sobely_mask = sobely_thresh(gray, sobel_kernel=7, thresh=(1, 100))
    channel = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 2]
    channel_mask = channel_thresh(channel, thresh=(150, 255))
    combined_mask = (channel_mask | sobelx_mask) & sobely_mask
    return combined_mask
```

To demonstrate the two steps above, here's the result of distortion correction (left column) and thresholding (right column) on all the test images:

![alt text][image2]

### 3\. Perspective transform

The code for my perspective transform includes a function called `warp_matrix()`, which appears in lines 101 through 122 in the file `lane_tracking.py`. The `warper_matrix()` returns the perspective transformation matrix from its hardcoded `src` and `dst` points.

```python
src = np.float32([[230, 700],
                  [1060, 700],
                  [687, 450],
                  [593, 450]])
dst = np.float32([[300, 720],
                [900, 720],
                [900, 50],
                [300, 50]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

### 4\. Lane line fitting

In order to fit the lane line, two steps were taken.

1. Identify the lane line pixels from the binary image obtained by thresholding and perspective transform. The task was accomplished by two functions:

  - `initial_lane_search(binary_warped)` performs walking window based lane line search. It is used on the first frame in a video, where no previous lane record was available.
  - `lane_track(binary_warped, best_xleft, best_xright)` takes previously found lane pixels `best_xleft` and `best_xright` as reference, and return lane pixels on the new frame in proximity as lane pixels.

2. Polynomial on the identified lane pixels. This is done in function `robust_fit(leftx, lefty, rightx, righty, best_xleft=None, best_xright=None)`. In this function, I tried to fit the left and right edge of the lane simultaneously based on the assumption that **the left and right edge of the lane should share the same polynomial coefficients upto a constant**. The return value of the fitting is a ndarray with 4 elements [c0, c1, c2, c3]. The left edge of the lane is approximated by `c[0]*y**2 + c[1]*y +c[3]`, while the right edge of the lane takes the form of `c[0]*y**2 + c[1]*y +c[2] + c[3]`. The function takes advantage of `RANSACRegressor` in `scikit-learn` to reject the noise pixels. To make the fitting more robust, I run RANSAC fitting for 10 times, and select the result closest to previous lane line record.

#### 5\. Lane curvature and offset calculation

Lane curvature and offset was calculated in lines #368 through #378 in `lane_tracking.py`. The conversion factor between x/y pixels and physical distance was measured on one frame in the video, then hardcoded.

#### 6\. Lane marking

I implemented this step in lines #431 through #438 in my code in `lane_tracking.py`. The area between the fitted left and right edge was filled and perspective transformed back and overlaid with the original image.

Here is the demonstration of the three aforementioned steps on the test images. Left column shows the original image, center column shows lane fitting, and right column shows lane marking and curvature / offset calculation.

![alt text][image4]

--------------------------------------------------------------------------------

## Pipeline (video)

Here's the result of running the pipeline on the project video.

<img src="./output_images/video_output.gif" width="720"/>


Here's a [link to the video as mp4](./output_images/video_output.mp4)

--------------------------------------------------------------------------------

## Discussion

One problem of my current solution is my fitting function sometimes doesn't perfectly fit the lane line. This often happens on one side of the edge. The reason of this problem was I enforced both edges to have the same polynomial coefficient except for the constant term. This would be a valid assumption if the road is perfect flat. But in the video, the road has some slope, although very shallow, making the lane lines appears to be deformed. As a result, the lane edges aren't always parallel. If I were going to pursue this project further, I'd adjust the fitting algorithm to encourage the left and right lane edge to have similar fitting coefficients, and allow them to have a small deviation.  


[//]: # "Image References"
[image1]: ./output_images/fig1_cam_cal.png "Undistorted"
[image2]: ./output_images/fig2_undistort_threshold.png "Road Transformed"
[image3]: ./output_images/fig3_perspective.png "Binary Example"
[image4]: ./output_images/fig4_lane_marking.png "Warp Example"
