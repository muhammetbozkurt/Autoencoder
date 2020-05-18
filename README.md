# Autoencoder
Before started I did share dataset because I am not sure if I am allowed to share it. Because I got dataset from a challenge. 

I planed this repo first part of a series. In this paper first we talk about “What is Autoencoders” after that I explain how I build a CNN based autoencoder for sign images and share its results.

## What is Autoencoder?

  An autoencoder is an unsupervised machine learning algorithm which aims to reconstruct  input. Maybe it looks a bit odd when reading it first but algorithm does not simply copy input layer to output layer. As figure 1 represent, algorithm first reduces representation then reconstructs input from reduced representation. This process provides algorithm to capture useful properties in input data. A noteworthy feature of autoencoders is while getting compressed version of input data, noise in input data is removed.

  First part of model which is input layer to middle layer is named as encoder part and other part of algorithm which is middle layer to output layer is named  as decoder part. Encoded features of input is obtained from middle layer that is also referred as bottleneck.

![Figure 1](https://github.com/muhammetbozkurt/Autoencoder/blob/master/autoencoder_rep.png)
###### *Figure 1: A basic autoencoder with a single layer (Charu C. Aggarwal, Neural Networks and Deep Learning, p.73)*


## Building Our CNN Autoencoder model

Building model I first directly used VGG-16 architecture as encoder part (threfore reverse it for decoder part). However, it had 31,470,529 trainable parameters which does not provide efficiency in training phase and also to train such a network need huge amount of data* I chanced it a bit. you can check it end of page (figure 2). 

In last version model have 308,593  trainable parameters which looks feasible. Moreover, model is so deep gradient vanishing problem can be occurred. To prevent gradient vanishing problem, batch normalization layers were added and also relu activation function were used in hidden layers. 

A data generator function is coded for generating noise unique data in training phase which described in details below. Every epoch 32 * 10000 data point will be generated.

After preprocess (detailed below) we have a binary structure in our data so sigmoid is output  laye activation function and binary crossentropy is our loss function.

**rule of tumb is the total number of training data points should be at least 2 or 3 times larger than number of parameters in neural networks*

##Preporcess

Sign images in our dataset only contains 0 or 255 as their pixels  values and every image has  different definations.

1. Resize image to (224, 224) using cv2.INTER_NEAREST as interpolation technique to preserve the binary structure in images.

2. Divide every pixels to 255 and Convert them float type.

After these steps pixels values are 1 or 0.

##Data Generation

data_generator funtion first reads images from training dataset then flipping and rotating images (or not) at the end adding pepper input image due to add noise. As we mentioned the main purposes of autoencoders are removing noise in training data and dimentionality reduction. Therefore adding noise is important for our autoencoder to gain ability to remove noise. 

Some examples of generated data:

![gen](https://github.com/muhammetbozkurt/Autoencoder/blob/master/gen1.png)
![gen 1](https://github.com/muhammetbozkurt/Autoencoder/blob/master/gen2.png)
![gen 2](https://github.com/muhammetbozkurt/Autoencoder/blob/master/gen3.png)

Results of trained model:

(leftside input, rightside output)


![sample](https://github.com/muhammetbozkurt/Autoencoder/blob/master/sample1.png)
![sample](https://github.com/muhammetbozkurt/Autoencoder/blob/master/sample2.png)
![sample](https://github.com/muhammetbozkurt/Autoencoder/blob/master/sample3.png)
![sample](https://github.com/muhammetbozkurt/Autoencoder/blob/master/sample4.png)
![sample](https://github.com/muhammetbozkurt/Autoencoder/blob/master/sample.png)
 
 
 
 ### Architecture:
 
 ![architecture](https://github.com/muhammetbozkurt/Autoencoder/blob/master/model_plot.png)
 ###### *Figure 2:  Model*
