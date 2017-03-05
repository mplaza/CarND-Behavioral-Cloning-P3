#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the nvdia autonomous vehicle team's architecture with some minor modifications.

The images were cropped to make the data train faster by elliminating the parts of the image that weren't relevant to the car driving (line 36) and the data was normalized using a Keras lambda layer (line 39).

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 with (model.py lines 42-46) 

The model includes RELU layers to introduce nonlinearity (code line 42-46).

There are four fully connected layers with one dropout layer to reduce overfitting (48-52).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (line 50).
The number of epochs were reduced and the model was trained and validated on different data(line 58). 

Data was also added where the car was driven counter clockwise around the lap to reduce the left turn bias and make the model more generalizable.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 55).

####4. Appropriate training data

I used center lane driving the vehicle around the lap several times, recovering from the left and right sides of the road and driving the vehicle counter clockwise around the track.


###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with a convolution neural network similar to the one used by the nvdia autonomous vegicle team. This seemed appropriate since the goal was to keep the vehicle on the road in the simulator, so the types of feature extractions would be very similar. The lenet network worked well for the last project, but it seemed like a good idea to go with a more powerful model for this more complex task.

I cropped the images so they would be easier to train and because in the first project this made it much easier to find the lane lines in the images. I also normalized the rgb values.

I split the data in to a training and validation set to see how well the model was working and it worked pretty well initially, but the validation error was high increased with high epochs.

To combat overfitting I introduced a dropout layer and decreased the number of epochs. 

I also noticed that the car wasn't driving super well and was driving to the left off the road, so I also collected more training data and included recovering from the left and right and driving counterclockwise around the track. The car was still having trouble around the steep turns so I began using the left and right camera images also with a correction added to their steering angle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x310x3 rgb image   						| 
| Cropping         		| 160x310x3 rgb image, outputs 72x320x3  		| 
| Lambda Normalization  | 										 		|
| Convolution 5x5		| 2x2 stride, VALID padding, outputs 34x158x24	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride,  VALID padding,outputs 15x77x36	|
| RELU					|												|
| Convolution 5x5 	    | 2x2 stride, VALID padding, outputs 6x37x48	|
| RELU					|												|
| Convolution 3x3 	    | 1x1 stride, VALID padding, outputs 4x35x64	|
| RELU					| 	        									|
| Convolution 3x3 	    | 1x1 stride, VALID padding, outputs 2x33x64	|
| RELU					| 	        									|
| Flatten				| output 4224									|
| Fully Connected 		| output 100									|
| Dropout 				| keep prob 0.5									|
| Fully Connected		| output 50										|
| Fully Connected 		| output 10										|
| Fully Connected 		| output 1										|

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center. The center lane driving didn't go off the side of the road, so this should teach the vehicle to recover to the center if it does.

I also drove counterclockwise on the track to augment the data and make it more generalized/ reduce the left turn bias.

After collecting this data and running the model, it performed well but was still now able to handle the steep left turn, so I also flipped the images to fully balance the r/l of the dataset and also used the right and left cameras to help the car correct itself when off to the side.

After the collection process, I had 36566 number of data points using the training data augmented with some images from left right and center cameras and flipped images. I then preprocessed this data by cropping and normalizing the images.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as this allowed the training accuracy to increase without reducing validation accuracy. I used an adam optimizer so that manually training the learning rate wasn't necessary.
