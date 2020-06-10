# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/TrainingDistribution.png "Training Distribution"
[image2]: ./examples/TrainingDatasetVisualization.png "Training Dataset Visualization"
[image3]: ./examples/RandomImageAugmentation.png "Random Image Augmentation"
[image4]: ./examples/InternetImages.png "Traffic Signs from internet"
[image5]: ./examples/AugmentedDataset.png "Augmented Training Dataset distribution"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/LeZest/sdc_pj1/blob/master/CarND-Traffic-Sign-Classifier-Project/ColabNotebook_Traffic_Sign_Classifier.ipynb)

Note that this is a **Colab notebook**. 
I uploaded the files in my Google drive and mounted the files in drive as follows: 
```python
from google.colab import drive
drive.mount('/content/gdrive')
```

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training, test & validataion data is distributed. 

![Training Distribution][image1]

Here is a visualization of random images from the dataset.

![Randon data in training dataset][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

It was clear from histogram that several classes is significantly underrepresented in the training set, which might lead to low accuracy on those classes and overfitting on the classes with higer than mean representation. 

I generated additional data by randomly adding any of the following attributes to a random image in each class until they have minimum of 809 images (original mean images of training dataset) in each calss: 

- Random gaussian blurr
- Random gaussian noise
- Random shift
- Combination of random blurr & random shift
- Combination of random noise & randoom shift

Here is the distribution after training set augmentation: 
![Training distribution after augmentation][image5]

After the dataset augmentation, I had a total of 46714 images in the dataset. Next I normalized all images to prep them for training. Here is an example of a set of normalized original image & a possible combination of augmented images (also normalized):

![Image Augmentation functions][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used LeNet model as reference and modified the top fully connected layer to contain 43 classes.  
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Grayscale       		| 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| Relu					|												|
| Dropout				|0.95											|
| Max pooling			| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| Relu					|												|
| Dropout				|0.85											|
| Max pooling			| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| outputs 400 									|
| Fully connected		| outputs 120 									|
| Relu					|												|
| Fully connected		| outputs 84 									|
| Relu					|												|
| Fully connected		| outputs 43   									|
| Softmax				|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters: 
EPOCHS = 70
BATCH_SIZE = 128
RATE = 0.001
BETA = 0.001 (for L2 regularizer)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used LeNet architecture as reference and noticed that it could reach a validation accuracy of upto 89% in German Traffic sign predictions. This is obviously below the target 93% validation accuraccy. Hence I added the following steps in the LeNet model & training pipeline:

- Droppout layer after Convolution layer 1, 0.95
- Dropout layer after Convolution layer 2, 0.85
- L2 regularizer with Beta = 0.001. Regularizer loss was calcualted by adding the loss of each layer
- Lastly to avoid overfitting, once the vaidation accuraccy had reached 91%, I reduced the learning rate from 0.001 to 0.0001 

The accuracy on my final model were:

validation set accuracy of 0.930
test set accuracy of 0.922


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Traffic signs from internet][image4]

I believe that the images that I have chosen are relatively easy to classify. They seem bright enough and easily distinguishable. Hence in my opinion, they should result in higher accuracy than the test images themselves. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					|Class ID		|Prediction ID 	| 
|:---------------------:|:-------------:|:-------------:| 
| 30 km/h      			|1 				| 1 			| 
| 70 km/h      			|4 				| 4 			|
| No entry     			|17				| 17 			|
| STOP    				|14				| 14			|
| Yield					|13				| 13			|


The model was able to correctly predict all 5 German traffic signs, which gives an accuracy of 100%, which is better than the accuracy on test images. However on testing the same images multiple times with the model, the softmax probabilities changes significantly and at times there were 4 out of 5 correct predictions. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the softmax outputs/probabilities of each test image: 

Image 1: **30km/h**
| Probability	|Prediction ID	|Prediction 	| 
|:-------------:|:-------------:|:-------------:| 
| 0.482    		|1 				| 30km/h 		| 
| 0.349    		|0 				| 20km/h 		|
| 0.119   		|5				| 80km/h 		|
| 0.049    		|4				| 70km/h		|
| 0.0002		|6				| End of 80km/h	|

Interesting to see that all the top 5 predictions were related a speed limit range indications and that the top 2 probabilities 30km/h & 20km/h probabilities are close. 

For the 2nd to 5th images, the probability for the 1st class id was above 99%. As example, here are softmax probability of image 4 (**STOP sign**)

Image 4: STOP
| Probability	|Prediction ID	|Prediction 		| 
|:-------------:|:-------------:|:-----------------:| 
| 0.999    		|14 			| STOP  			| 
| 0.000   		|33 			| Turn right ahead	|
| 0.000   		|17				| No Entry 			|
| 0.000    		|34				| Turn right ahead	|
| 0.000 		|38				| Keep Right		|



