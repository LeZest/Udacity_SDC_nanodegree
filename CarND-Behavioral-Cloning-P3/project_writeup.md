# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_summary.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* project_report.md summarizing the results
* video.mp4 demonstrating a video captured using model.h5 on track 1 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

As recommended in the lecture videos, I implemented NVidia's CNN for End to End training (model.py line 120). 
The data is first normalized using a keras lambda layer and the images are then cropped in vertical direction from top (75 lines) and bottom (20 lines) using keras cropping 2d layer (model.py line 124). 
I also added RELU activation functions in all 5 convolutional layers to introduce non-linearity. 

#### 2. Attempts to reduce overfitting in the model

A dropout layer with a factor of 0.25 was added to the first dense layer in order to avoid overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 151).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started off by implementing the nVidia's End to End lerning CNN architecture which I believe provides a perfect reference to start training the model. I trained this model on the track 1 data provided by Udacity. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added a dropout layer after Dense_1 layer with a dropout factor of 0.25. 
Then I added RELU activation functions in the convolutional layers to add non linearity. 

The final step was to run the simulator to see how well the car was driving around track one. I relaized that the vehicle would not driver correctly near the bridge and on sharp left & right turns. Hence I decided to retrain the model with additional data by loading existing models (models.py line 144). This allowed me to finetune the model everytime I saw an error in the driving behavior of the car in Autonomous mode. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 120-140) consisted of a convolution neural network with the following layers and layer sizes: 

![Model Summary][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded 4 laps on track one using center lane driving. Since I had also noticed that the model had significant left turn bias, I added 4 laps of reverse driving data on track 1 as well. 

The car was still not able to drive around the bridge area below. So I decided to record data only around the bridge by drive through it in both directions. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the corners. 

To augment the dataset, I also flipped images and added offset to the images from left & right camera
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained for 5 epochs for the training as well as for finetuning the model with additional data. 
