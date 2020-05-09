# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[cnn]: ./images/cnn.png "Model Visualization"
[flip]: ./images/flip.png "Flipping"
[crop]: ./images/crop.png "Crop"
[allview]: ./images/allview.png "All view of a image"
[improve]: ./images/improve.png "From left to center"
[center]: ./images/center.jpg "on center"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 59-80) 

The model includes RELU layers to introduce nonlinearity (code line 62), and the data is normalized in the model using a Keras lambda layer (code line 59). 

#### 2. Attempts to reduce overfitting in the model

After training for 5 epochs, I saw training loss of 0.0154 and validation loss of 0.018. This means the model is not suffering from overfitting. Intially I had tried different architecture and I had included dropout layers. But in this architecture dropout layer didn't do any good to the model. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 82).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The model was not doing well on my data alone so I concatinated the sample data provided with this project and then it really worked well.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet but with more filters from 16 to 64. I thought of starting with something simple and test whether my training data is good.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include dropout layers.

But it didn't improve, the model was not doing good in the simulator. Then I implemented the Nvidia's CNN architecture alongwith the provided sample data and then only the model started working pretty well.

The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 59-80) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][cnn]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded first lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to center even if it get away from it somehow. These images show what a recovery looks like starting from left towards center:

![alt text][improve]

To augment the data sat, I also flipped images and angles thinking that this would give the model new perspective. For example, here is an image that has then been flipped:

![alt text][flip]
I also included all left, center and right views of images with slight change in angle of 0.25 radian. Here is an image from all views:

![alt text][allview]


I also cropped images as the upper part of image is not really playing any role for the model improvement. For example, here is an image that has then been cropped:

![alt text][crop]

After the collection process, I then preprocessed this data by centering around zero with small standard deviation using lambda layer of keras.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by accurary of the model. I used an adam optimizer so that manually training the learning rate wasn't necessary.
