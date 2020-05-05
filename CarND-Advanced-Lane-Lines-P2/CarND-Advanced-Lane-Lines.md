## Advanced Lane Lines write-up

### This writeup explains the pipeline implemented in CarND-Advanced-Lane-Lines.ipynb. 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: output_images/undistorted_chessboard.png "Undistorted Chessboard"
[image1a]: output_images/undistorted_highway.png "Undistorted Highway"
[image2]: output_images/Perspective_Transform_straight.png "Road Transformed Straight Lane"
[image2a]: output_images/Perspective_Transform_curved.png "Road Transformed Curved Lane"
[image3]: output_images/binary_combo_example.png "Binary Example"
[image4]: output_images/warped_binary.png "Warped Binary Example"
[image4a]: output_images/histogram_warped.png "Warped Binary Example"
[image5]: output_images/color_fit_lines.png "Fit Visual"
[image5]: output_images/color_fit_lines.png "Search around polynomial function"
[image6]: output_images/example_output.png "Output"
[video1]: test_videos_output/project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the IPython notebook "CarND-Advanced-Lane-Lines.ipynb" [Compute the camera calibration matrix and distortion coefficients](#Compute-the-camera-calibration-matrix-and-distortion-coefficients)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. and defined the function `cal_undistort()`: 

![Chessboard image][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I defined a function `cal_undistort` that takes `image`, `objpoints` and `imgpoints`  as inputs and outputs the Undistorted image. The code for this step is contained in the IPython notebook "CarND-Advanced-Lane-Lines.ipynb"
[Section title](#Test-distortion-correction-coefficients-on-a-raw-image)

![Undistorted Highway][image1a]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this section is written in IPython notebook "CarND-Advanced-Lane-Lines.ipynb"[Combine gradients & S Color thresholds to create a thresholded binary image.](#Combine-gradients-&-S-Color-thresholds-to-create-a-thresholded-binary-image.)

A function `combine_col_grad()` is defined to combine color transforms & gradients to create thresholded binary image. 
```python
    colors_gradients = combine_col_grad(image, s_thresh=(0, 255), sob_thresh = (20,100), mag_thresh=(30,100), dir_thresh=(0.7, 1.3)) #
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, sob_thresh=(sob_thresh[0], sob_thresh[1])) # Sobel X gradient
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, sob_thresh=(sob_thresh[0], sob_thresh[1])) # Sobel Y gradient
    mag_binary = mag_threshold(image, sobel_kernel=9, mag_thresh=(mag_thresh[0], mag_thresh[1])) # Magnitude gradient
    dir_binary = dir_threshold(image, sobel_kernel=15, dir_thresh=(dir_thresh[0], dir_thresh[1])) # Direction Gradient
    s_binary = hls_select(image, thresh=(90, 255)) # S Color gradient
```

Here's an example of my output for this step. 
![Combined S-Color & Gradient Thresholded Binary][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this section is written in IPython notebook "CarND-Advanced-Lane-Lines.ipynb"[Perspective transform](#Perspective-transform)
The `dewarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I used an image with straight lane lines to determine `src` co-ordinates and manually fitting the `src` to `dst`. Here is my `src` & `dst` values.  

```python
sl_bot = [274,670]
sl_top = [577,460]
sr_bot = [1042,670]
sr_top = [708,460]

xsize =  undist.shape[1] #undistorted image
ysize =  undist.shape[0] #undistorted image
xwarp = 280

dl_bot = [xwarp,ysize]
dl_top = [xwarp,0]
dr_bot = [xsize-xwarp,ysize]
dr_top = [xsize-xwarp,0]

src = np.float32([sl_bot,sl_top,sr_bot,sr_top])
dst = np.float32([dl_bot,dl_top,dr_bot,dr_top])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 274, 670      | 280, 720      | 
| 577, 460      | 280,   0      |
| 1042, 670     | 1000, 720     |
| 708, 460      | 1000, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Sample Image with straight lane lines][image2]

![Warped image with curved lane lines][image2a]

I also converted the combined threshold image into **warped binary image**, which was then input into the next step to detect lane lines.  
![Perspective transform of the combined threshold image][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this section is written in IPython notebook "CarND-Advanced-Lane-Lines.ipynb"[Detect lane pixels and fit to find the lane boundary](#Detect-lane-pixels-and-fit-to-find-the-lane-boundary.) 

- defined a `hist()` function to calculate histogram of the **warped binary image**. Histogram with the highest peak on each side of the midpoint is taken as left & right lane and used as reference for detecting left and right lane lines in the next step. 
- defined `find_lane-pixels()` which takes **warped binary image** as input and used the sliding window approach to detect the active pixels in left and right lines and returns the coordinates **leftx, lefty, rightx, righty** 
- defined `fit_polynomial()` which takes the coordinates from previous steps as input and determines the best fit polynomial of 2nd order and plots on the **warped binary image** 

Confirmed that the function implementation is working fine and tested on the **warped binary image** from previous step

![Histogram of warped binary image][image4a]
![Polynomial fit in yellow and colored Right and left lane pixels][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this section is written in IPython notebook "CarND-Advanced-Lane-Lines.ipynb"[Curvature & Vehicle position](#Determine-the-curvature-of-the-lane-and-vehicle-position-with-respect-to-center.)

- defined `measure_curvature_pixels() & measure_curvature_real()` to calculate curvature of left & right lane and then took an average of the same in number of pixels & 'meters'. 
- defined `vehicle_offset()` to calculate the distance in 'cm' by calculating the difference between center of the image and center of the lane line using the polynomial fit function. 
 
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this section is written in IPython notebook "CarND-Advanced-Lane-Lines.ipynb"[Plot lanes on the original image](#Plot-lanes-back-on-the-original-image).  Here is an example of my result on a test image:

- Defined a `plot_poly_window()` to overlay the output of step 4 & 5 on the original image:

![Final Result][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The code for this section is written in IPython notebook "CarND-Advanced-Lane-Lines.ipynb"[Pipeline](#Video-Pipeline)

Here's a [link to video output][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The sliding window approach failed to detect lane lines in some scenarios where vehicle is under a bridge. We could add an averaging function inside polynomial calculation to improve lane detection accuracy. 
- The pipeline doesnt work on sharp turns since the polynomial fit function fails. One idea could be to dynamically change the MARGIN of window depending upon the curvature in previous frame. 
