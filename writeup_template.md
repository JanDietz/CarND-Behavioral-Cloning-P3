
[//]: # (Image References)

[image1]: ./examples/center_2018_01_01_14_38_50_034.jpg "Model Visualization"
[image2]: ./examples/center_2018_01_01_14_38_50_034.jpg "Center"
[image3]: ./examples/left_2018_01_01_14_38_50_034.jpg "left"
[image4]: ./examples/center_2018_01_01_14_38_50_034.jpg "center"
[image5]: ./examples/right_2018_01_01_14_38_50_034.jpg "right"
[image6]: ./examples/center_2018_01_01_14_39_09_375.jpg "Normal Image"
[image7]: ./examples/center_2018_01_01_14_39_09_375_spiegelt.jpg "Flipped Image"

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

I chose the Nvidia architecture due of its better performance then the LeNet architecture. I consists of Normalization layer and a cropping layer. General a network performs better, when data is normalized around zero and cropped a part of the image where no road is seen. 

Then there are 5 convolutional layers with a filter depth increase: 3 of them have a 5x5 filter size and a 2x2 stride, the last 2 convultions layers are only 3x3 filters with a 1x1 stride. 

After a Flatten layer there are 4 fully connected layers with a final output of 1 for the steering angle.

#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layer due it is not necessary. The model is trained with 5 epochs and the validation loss was decreasing continously.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the LeNet Implementation. But it turned out some weak spots in the section with missing right lane line.
Therefore I used the Nvidia architecture which worked fine for me.

I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
To combat the overfitting, I added a dropout layer. That reduced the gap.

Most of my time I spend on preprocessing the images. I felt most important was to use the left and right camera images with an moddified steering angle with the result drives more right when its left and vice versa.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded only one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I took the left and right camera images for training. So the model knows what to do in these positions:

![alt text][image3]
![alt text][image4]
![alt text][image5]

I did no recourd track two, because it was very different and the aim was to pass track 1.

To augment the data sat, I also flipped images and angles thinking that this would help to generalize the model. So the car would be able to drive counter clockwise as well. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection and preprocessing , I had 6100 number of data points that feeds the model.
I finally randomly shuffled the data set and put 20% of the data into a validation set. The 80% of the data were into the training set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 to 5.
