

# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/11.jpg "Original"
[image2]: ./examples/22.jpg "Grayscaling"
[image3]: ./examples/arc.png "Grayscaling"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---


## Model Architecture and Training Strategy

![alt][image3]

### 1. LeNet Architecture
The architecture I have used to train my model is LeNet architecture. I have used this architecture because it is simple and easy to build. Also, it is not complicated network so the training will not take too much time.  

### 2. LeNet Structure 

The netowrk is made of two convlutional layers,  and pooling layers that connected to a flatten layer. The flatten layer is then followed by two fully connected layers. The lenet architecture takes 32 by 32 input however, our images are (160,320) but keras will take care of the input shape. 

The first convlutional layer produces 6 features each with  size of 28 by 28 and 5 by 5 filters.  The next layer is maxpooling layer with 6 feature map of size 14 by 14 then another convlutional layer with 16 features map of size 5 by 5 and another maxpooling layer then two fully connected layers of size 120 and 84 with last fully connected layer with one node only to predict the steering angle.


 **Note**: *The whole structe is implemented in line 58 in the function "Lenet"*

**Note**: The two fully connected layers are followed by softmax layer, but becaue the problem here is regression we didn't include it. 


### 3. Training Data Set

I have trained the data several times with no sucess in getting the car to stay in the middle of the road. The fix I have applied to sort this issue out is to crop, normlaize and augment the data. I have used a correction factor and both left,center and right view to correct the steering angle each time the car slides off the road. In augmentation, I have flipped the data so the car can see diffrent perspective (both for driving on the right and left sides) and I have reversed the steering angle. In cropping the image, I have removed the sky and the front of the car. The cropping was includded in the Keras model to make the process faster (over the GPU). 



**Note**: I have not recorded my own data because the simulator was lagging so I have used the data provided by Udacity Team. 

**Note**: The code for agumenting the images can be seen in the function "augment_images". Also,the cropping can be seen in Keras function "Lenet" line 74-76.

<center>

Original Image:

![alt][image1]

Image After Cropping : 

![alt][image2]

</center>

### 4. Reducing Overfitting 
I have not add dropout layer to reduce overfitting but I have minmized the number of epoch to 3 and I have splitted and shuffled the data to validate the accuracy of my model. This can be seen in "Lenet" function line 89. 


### 5. Paramter Tuning 
I have used adam optimized to tune my model. 



