# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Identify potential shortcomings of the solution and suggest possible improvements.

---

### Reflection

### 1. Describe the pipeline. 

The pipeline consists of following steps:

- **Canny edge detection**: Use the Canny Transform function to find edges on the image. Kernel size & thresholds were taken from Lesson 4: CV fundamentals as reference. 
- **Find region of interest**: Select the region of interest using **image size** i.e. **xsize** & **ysize** parameteres so that only the lane markings are selected for Hough Transform. 
- **Hough Image**: Use Hough Transform to identify Hough Lines. Output of this function is Hough Line co-ordinates & a blank image with Hough lines plotted on it. 
- **Seperate Lines**: The output of Hough Transform **lines** & **line_image** is provided as input to seperate the left & right lane markings. Eliminate any hough lines with horizontal slopes **|slope|< 0.3** and determine coordianates of the left & right lanes. Furthermore we also determine the top & bottom most coordinates of hough lines for both sides. 
-- **Extrapolate Lane**: Use **polyfit()** to determine left & right lane coordinates at the bottom of the image **ysize**. Then merge these coordinates with the left & right lane coordinates.   
- **Plot lines**: Plot the lane markings from **seperate_lines()** on an empty image of the same size as input image.
- **weighted_img**: Merges the output of **plot_lines()** with the original image to represent the lines on it.

### 2. Identify potential shortcomings with your current pipeline

Potential shortcomings would be the following:

- **Curved Lines**: Since we are plotting straight lines on the image, any curves on the road cannot be accurately traced. 
- **Images with Tree shadows**: It was visible from the output of **challenge video** that when many edges are detected around the lane markings, for example as a result of tree shadows on road, the pipeline is unable to detect lane markings with enough accuracy. 


### 3. Suggest possible improvements to your pipeline

- **Combine multiple lines**: Instead of plotting one straight line on each side, the lane can be broken into multiple lanes joined with each other (depending upon slope) so that a curved line can be traced with improved accuracy. Otherwise a non linear polynomial function can also be used. 
- **Use average over previous lanes**: Abrupt changes in lane slopes can be avoided if we use average slope of lanes from previous 2 or 3 images. Thus changes in lane slopes due to multiple edges around the lane markings can be avoided. 

