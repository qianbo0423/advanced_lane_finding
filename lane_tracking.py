import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import glob
from sklearn import linear_model
from moviepy.editor import VideoFileClip, ImageSequenceClip


def camera_calibration():
    objpoints = []
    imgpoints = []
    images = glob.glob('camera_cal/calibration*.jpg')
    for fname in images:
        if fname == 'camera_cal/calibration1.jpg':
            nx = 9
            ny = 5
        else:
            nx = 9
            ny = 6

        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        if fname == 'camera_cal/calibration5.jpg':
            objp = np.delete(objp, (0), axis=0)

        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (1280, 720), None, None)
    return mtx, dist


def sobelx_thresh(img, sobel_kernel=5, thresh=(10, 200)):
    """
    apply gradient thresholding along x axis on a gray scale iamge
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(abs_sobelx / np.max(abs_sobelx) * 255)
    binary_sobel = np.zeros_like(img)
    binary_sobel[(scaled_sobelx > thresh[0]) &
                 (scaled_sobelx <= thresh[1])] = 1
    return binary_sobel


def sobely_thresh(img, sobel_kernel=5, thresh=(10, 200)):
    """
    apply gradient thresholding along y axis on a gray scale iamge
    """
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)
    scaled_sobely = np.uint8(abs_sobely / np.max(abs_sobely) * 255)
    binary_sobel = np.zeros_like(img)
    binary_sobel[(scaled_sobely > thresh[0]) &
                 (scaled_sobely <= thresh[1])] = 1
    return binary_sobel


def gradient_dir_thresh(img, sobel_kernel=5, thresh=(0.7, 1.3)):
    """
    apply threshold on direction of gradient on a gray scale iamge
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    angle = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_sobel = np.zeros_like(angle)
    binary_sobel[(angle > thresh[0]) & (angle <= thresh[1])] = 1
    return binary_sobel


def gradient_mag_thresh(img, sobel_kernel=5, thresh=(10, 200)):
    """
    apply threshold on magnetude of gradient on a gray scale iamge

    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = np.uint8(mag / np.max(mag) * 255)
    binary_sobel = np.zeros_like(mag)
    binary_sobel[(mag > thresh[0]) & (mag <= thresh[1])] = 1
    return binary_sobel


def channel_thresh(img,  thresh=(10, 200)):
    """
    apply threshold on value of a single channel
    """
    binary_channel = np.zeros_like(img)
    binary_channel[(img > thresh[0]) & (img <= thresh[1])] = 1
    return binary_channel


def color_gradient_thresh(img):
    """
    combined thresholding on a colored image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2]
    sobelx_mask = sobelx_thresh(gray, sobel_kernel=7, thresh=(25, 120))
    sobely_mask = sobely_thresh(gray, sobel_kernel=7, thresh=(1, 100))
    channel = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 2]
    channel_mask = channel_thresh(channel, thresh=(150, 255))
    combined_mask = (channel_mask | sobelx_mask) & sobely_mask
    return combined_mask


def warp_matrix(video=0):
    """
    calculate the transformation matrix for perspective transform
    """
    if video == 0:
        src = np.float32([[230, 700],
                          [1060, 700],
                          [687, 450],
                          [593, 450]])
        # src = np.float32([[340, 700],
        #                   [1180, 700],
        #                   [702, 450],
        #                   [607, 450]])
    elif video == 1:
        src = np.float32([[320, 700],
                          [1090, 700],
                          [670, 450],
                          [636, 450]])
    dst = np.float32([[300, 720],
                      [900, 720],
                      [900, 50],
                      [300, 50]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def mark_clusters(img):
    X = np.transpose(np.nonzero(img))
    labels = dbs.fit_predict(X)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    class_ct = 0
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        if class_member_mask.astype(int).sum() > 200:
            class_ct += 1
            xy = X[class_member_mask]
            plt.plot(xy[:, 1], xy[:, 0], '.',
                     markerfacecolor=tuple(col), markersize=3)
    print(class_ct)
    plt.imshow(img, cmap='gray')


def initial_lane_search(binary_warped):
    """
    search driving lanes on a warped binary image using the walking window algorithm
    only run this function on the first frame of a video
    return value: leftx, lefty, rightx, righty

    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(
        binary_warped[int(binary_warped.shape[0] / 3):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
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

    left_shift = np.array([0])
    right_shift = np.array([0])

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
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_last = leftx_current
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            np.append(left_shift, (leftx_current - leftx_last))
        else:
            leftx_current += left_shift[-3:].mean()
        if len(good_right_inds) > minpix:
            rightx_last = rightx_current
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            np.append(right_shift, rightx_current - rightx_last)
        else:
            rightx_current += right_shift[-3:].mean()

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def lane_track(binary_warped, best_xleft, best_xright):
    """
    track the previously found driving lane on new frames
    binary_warped: binary image after thresholding and perspective transform
    best_xleft: left driving lane found in previous frames
    best_xright: right driving lane found in previous frames
    """
    # Choose the number of sliding windows
    nwindows = 20
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Set the width of the windows +/- margin
    margin = 100
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_y_mid = int((win_y_low + win_y_high) / 2)
        leftx_current = best_xleft[win_y_mid]
        rightx_current = best_xright[win_y_mid]
        win_xleft_low = int(leftx_current - margin)
        win_xleft_high = int(leftx_current + margin)
        win_xright_low = int(rightx_current - margin)
        win_xright_high = int(rightx_current + margin)
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # plt.imshow(out_img)
    return leftx, lefty, rightx, righty


def robust_fit(leftx, lefty, rightx, righty, best_xleft=None, best_xright=None):
    """
    fit the left and right driving lane simutaneously using RANSAC.
    Assuming left and right driving lane share the same polynomial coefficents upto a constant
    run RANSAC 10 times and select the result closest to the previouly found lanes
    leftx, lefty, rightx, righty: [ndarray] x and y coordinate of left/right driving lane to be fitted
    best_xleft, best_xright: [ndaray] previously found driving lane
    return: [c] coefficent for left and right lane
    to recover left lane:
    x = c[0]*y**2 + c[1]*y + c[3]
    to recover right lane
    x = c[0]*y**2 + c[1]*y + c[2] + c[3]
    """
    Y_left = np.stack([lefty**2, lefty, np.zeros_like(lefty)], axis=1)
    Y_right = np.stack([righty**2, righty, np.ones_like(righty)], axis=1)
    Y = np.concatenate((Y_left, Y_right))
    x = np.concatenate((leftx, rightx))
    weight = np.concatenate((np.ones_like(leftx), np.ones_like(rightx)
                             * (len(leftx) / len(rightx))))
    ransac = linear_model.RANSACRegressor(max_trials=100)
    if best_xleft is None:
        ransac.fit(Y, x, weight)
        k = ransac.estimator_.coef_
        b = ransac.estimator_.intercept_
        return np.append(k, b)
    else:
        err = []  # keep track of the lane deviation of the trails
        coeff = []
        y = np.arange(len(best_xleft))
        for i in range(10):
            ransac.fit(Y, x, weight)
            k = ransac.estimator_.coef_
            b = ransac.estimator_.intercept_
            left_err = ((k[0] * y**2 + k[1] * y + b - best_xleft)**2).mean()
            right_err = ((k[0] * y**2 + k[1] * y +
                          k[2] + b - best_xright)**2).mean()
            err.append(left_err + right_err)
            coeff.append(np.append(k, b))
        best_try = np.argmin(np.array(err))
        return coeff[best_try]


def annotate_lane(coeff, img_shape=(720, 1280)):
    """
    draw the fitted driving lane on binary warped image
    """
    img = np.zeros([img_shape[0], img_shape[1], 3], dtype=np.uint8)
    lane_y = np.arange(0, img_shape[0])
    left_lane_x = coeff[0] * lane_y**2 + coeff[1] * lane_y + coeff[3]
    right_lane_x = coeff[0] * lane_y**2 + \
        coeff[1] * lane_y + coeff[2] + coeff[3]
    pts_left = np.array([np.transpose(np.vstack([left_lane_x, lane_y]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_lane_x, lane_y])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(img, np.int_([pts]), (0, 255, 0))
    return img


def calculate_curvature(c, mpx=0.0059, mpy=0.042):
    xd = 2 * mpx / (mpy**2) * c[0] * 350 + c[1] * mpx / mpy
    xdd = 2 * mpx / (mpy**2) * c[0]
    curv = ((1 + xd**2)**1.5) / np.absolute(xdd)
    return curv


def calculate_lane_off_center(c, mpx=0.0059, mpy=0.042):
    y0 = 720
    offset_pixel = (c[0] * (y0**2) + c[1] * y0 + c[3] + c[2] / 2) - 1280 / 2
    return offset_pixel * mpx


class Lane(object):
    """
    class for tracking driving lanes in video
    """

    def __init__(self, video=0):
        self.frame_ct = 0
        self.detected = False
        self.recent_xfitted_left = []
        self.recent_xfitted_right = []
        self.recent_fitted_coeff = []
        self.best_xleft = None
        self.best_xright = None
        self.best_coeff = None
        self.current_fit = None
        self.cand_fit = None
        self.coeff_diff = np.array([0, 0, 0, 0])
        # camera calibration
        self.img_shape = None
        self.mtx, self.dist = camera_calibration()
        self.M, self.Minv = warp_matrix(video)
        self.img_masks = []
        self.antn_img = []
        # calculations
        self.current_curvature = None
        self.current_lane_offset = None

    def detect_lane_from_img(self, img):
        """
        apply the pipeline to find driving lanes in each frame
        """
        self.frame_ct += 1
        img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        img_mask = color_gradient_thresh(img)
        img_mask_warp = cv2.warpPerspective(
            img_mask, self.M, (img_mask.shape[1], img_mask.shape[0]), cv2.INTER_LINEAR)

        if self.detected == False:
            # run initial_lane_search on first frame
            self.img_shape = [img.shape[0], img.shape[1]]
            leftx, lefty, rightx, righty = initial_lane_search(img_mask_warp)
            self.cand_fit = robust_fit(leftx, lefty, rightx, righty)
        else:
            # track the lanes in following frames
            leftx, lefty, rightx, righty = lane_track(
                img_mask_warp, self.best_xleft, self.best_xright)
            self.cand_fit = robust_fit(leftx, lefty, rightx, righty,
                                       self.best_xleft, self.best_xright)

        self.update_history()  # update lane parameters
        # save the annotated images
        marking = annotate_lane(self.current_fit)
        img_mask_warp_ = np.dstack(
            [img_mask_warp, img_mask_warp, img_mask_warp]) * 255
        img_mask_warp_[[lefty, leftx]] = [255, 0, 0]
        img_mask_warp_[[righty, rightx]] = [0, 0, 255]
        self.img_masks.append(cv2.addWeighted(
            img_mask_warp_, 1, marking, 0.3, 0))

        marking_unwarp = cv2.warpPerspective(
            marking, self.Minv, (img_mask.shape[1], img_mask.shape[0]), cv2.INTER_LINEAR)
        img_marking_unwarp = cv2.addWeighted(img, 1, marking_unwarp, 0.3, 0)
        cv2.putText(img_marking_unwarp, 'CurvR={:.1f}m'.format(
            self.current_curvature), (700, 80), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), thickness=3)
        cv2.putText(img_marking_unwarp, 'Offset={:.2f}m'.format(
            self.current_lane_offset), (700, 160), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), thickness=3)

        self.antn_img.append(img_marking_unwarp)

    def update_history(self):
        y = np.arange(0, self.img_shape[0])
        if self.detected == False:
            # on first frame, initialize all parameters
            print('initial lane detection')
            self.current_fit = np.array(self.cand_fit)
            c = self.current_fit
            self.recent_xfitted_left = np.array(
                c[0] * y**2 + c[1] * y + c[3]).reshape(-1, 1)
            self.recent_xfitted_right = np.array(
                c[0] * y**2 + c[1] * y + c[2] + c[3]).reshape(-1, 1)
            self.recent_fitted_coeff = np.array(c).reshape(-1, 1)
            self.best_xleft = c[0] * y**2 + c[1] * y + c[3]
            self.best_xright = c[0] * y**2 + c[1] * y + c[2] + c[3]
            self.best_coeff = c
            self.detected = True
        else:
            # lane tracking
            c = np.array(self.cand_fit)
            cand_xleft = c[0] * y**2 + c[1] * y + c[3]
            cand_xright = c[0] * y**2 + c[1] * y + c[2] + c[3]
            if (((cand_xleft - self.best_xleft)**2).mean() < 2500) & (((cand_xright - self.best_xright)**2).mean() < 2500):
                # if the fitted lane on the new frame is close to previous frames within 50 pixels, it's a good fit
                # save the fit to history
                self.recent_xfitted_left = np.concatenate(
                    [self.recent_xfitted_left, cand_xleft[:, np.newaxis]], axis=1)
                self.recent_xfitted_right = np.concatenate(
                    [self.recent_xfitted_right, cand_xright[:, np.newaxis]], axis=1)
                self.recent_fitted_coeff = np.concatenate(
                    [self.recent_fitted_coeff, c[:, np.newaxis]], axis=1)
                # update the best fits to the mean value of last 10 fits
                self.best_xleft = self.recent_xfitted_left[:, -10:].mean(
                    axis=1)
                self.best_xright = self.recent_xfitted_right[:, -10:].mean(
                    axis=1)
                self.best_coeff = self.recent_fitted_coeff[:, -10:].mean(
                    axis=1)
                # use the mean of last 10 fits for current frame
                self.current_fit = self.best_coeff
            else:
                # if it's not a good fit, print a warning and use the saved best fit for current frame
                print('lane deviate from previous at frame {}'.format(self.frame_ct))
                print(((cand_xleft - self.best_xleft)**2).mean())
                print(((cand_xright - self.best_xright)**2).mean())
                self.current_fit = self.best_coeff
        self.current_curvature = calculate_curvature(self.current_fit)
        self.current_lane_offset = calculate_lane_off_center(self.current_fit)


if __name__ == "__main__":
    clip = VideoFileClip('project_video.mp4')
    clip = [frame for frame in clip.iter_frames()]
    lane = Lane(video=0)
    for img in clip:
        lane.detect_lane_from_img(img)
    gif_clip = ImageSequenceClip(lane.antn_img, fps=25)
    gif_clip.write_videofile('video_output.mp4')
