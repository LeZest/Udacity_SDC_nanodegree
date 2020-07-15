import math
import cv2
from IPython import display
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
###################################################################
# Read data files and arrange as training data
import csv
import imageio
#csv_path = '../../../opt/carnd_p3/data/driving_log.csv'
#img_path = '../../../opt/carnd_p3/data/IMG/'

#csv_path = 'data_20200705_1/driving_log.csv'
img_path = 'data_recovery_0707/IMG/'   # path to IMG folder (all images are copied in this folder)
data_path = ['data_recovery_0707/']    # list of paths to csv folder
# data_path = ['../../../opt/carnd_p3/data/', '../../../opt/carnd_p3/data_20200630/', '../../../opt/carnd_p3/data_20200704/', '../../../opt/carnd_p3/data_20200705/']
#data_path = ['../../../opt/carnd_p3/20200707_0/']
#img_path = '../../../opt/carnd_p3/20200707_0/IMG/'

# combine all 'driving_log.csv' files in data_path list to samples
samples = []
for filename in data_path:
    with open(filename + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for line in reader:
            samples.append(line)

# Split the samples into training & validation dataset.
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.20)
print ('length of traning samples is ', len(train_samples))
print ('length of validation samples is ', len(validation_samples))

# Use generator function to read only batch_size images in one iteration to prevent memory overload.
import sklearn
def generator(samples, batch_size=32):
    num_samples = len(samples)
    offset = 0.1  # offset steering angle to left and right images
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = img_path + batch_sample[0].split('/')[-1]
                center_image = imageio.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # flip center images
                image_flipped = np.fliplr(center_image)
                angle_flipped = center_angle*(-1.0)
                images.append(image_flipped)
                angles.append(angle_flipped)
                # Add steering angle offset to left & right images and then flip the images to ccreate additional dataset
                right_image = imageio.imread(img_path + batch_sample[1].split('/')[-1])
                right_angle = max((center_angle - offset),-1.0)
                images.append(right_image)
                angles.append(right_angle)                
                # Flip right image 
                #right_flipped = np.fliplr(right_image)
                #right_flipped_angle = right_angle*(-1.0)
                #images.append(right_flipped)
                #angles.append(right_flipped_angle)  
                
                left_image = imageio.imread(img_path + batch_sample[2].split('/')[-1])
                left_angle = min((center_angle + offset),1.0)
                images.append(left_image)
                angles.append(left_angle)
                # Flip left image 
                #left_flipped = np.fliplr(left_image)
                #left_flipped_angle = left_angle*(-1.0)
                #images.append(left_flipped)
                #angles.append(left_flipped_angle)  
                # Data augmentation: flip center images with abs(steering angle)>0.09 
                # if abs(center_angle) > 0.09:                 
                    # flip center image
                    # image_flipped = np.fliplr(center_image)
                    # angle_flipped = center_angle*(-1.0)
                    # images.append(image_flipped)
                    # angles.append(angle_flipped)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

##########################################################################################
# Define Model & train CNN
# Import Keras libraries
from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from sklearn.utils import shuffle
from keras.callbacks import CSVLogger

# Select if you want to train the model from scratch or use an existing model to train with new data.
#train = 'simple' # Choose in case you just want to check the pipeline
#train = 'nvidia' # Train a new model from scratch
train = 'retrain' # Fine-tune an existing model with additional data
# Model Pipeline
model = Sequential()

if train == 'nvidia':
    # Normalization
    model.add(Lambda(lambda x: (x/255.0 -0.5), input_shape=(160,320,3), name='Lambda_1'))
    # Crop image
    model.add(Cropping2D(cropping=((75, 20), (0,0)), name='Crop2D_1'))
    #5 Convolution layers
    model.add(Conv2D(24, kernel_size=5, strides=2, padding='valid', activation='relu', name='Conv2D_1'))
    model.add(Conv2D(36, kernel_size=5, strides=2, padding='valid', activation='relu', name='Conv2D_2'))
    model.add(Conv2D(48, kernel_size=5, strides=2, padding='valid', activation='relu', name='Conv2D_3'))
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu', name='Conv2D_4'))
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu', name='Conv2D_5'))
    # Flatten
    model.add(Flatten(name='Flatten_1'))
    # 4 fully connected layers
    model.add(Dense(100))
    model.add(Dropout(0.25))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    nepochs = 5
elif train == 'retrain': # load existing model to retrain with new data. 
    print ('Fine tuning with new data')
    nepochs = 3
    model = load_model('models/model_20200706_relu_off01_v4.h5')
else:
    model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3), name='Lambda_1'))
    # model.add(Lambda(lambda x: (x/127.5 - 1.0), input_shape=(160,320,3), name='Lambda_1'))
    #model.add(Flatten(input_shape = (160,320,3)))
    model.add(Flatten(name='Flatten_1'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')    
    nepochs = 5

model.summary()
csv_logger = CSVLogger('models/log_20200707_relu_off01_v1.csv', append=True, separator=';')

history_object = model.fit_generator(train_generator, steps_per_epoch= math.ceil(len(train_samples)/batch_size), validation_data= validation_generator, validation_steps= math.ceil(len(validation_samples)/batch_size), epochs=nepochs, verbose=1, callbacks=[csv_logger])

# save model
model.save("models/model_20200707_relu_off01_v1.h5")
print ('Model Saved')

'''
### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
##########################################################################################
