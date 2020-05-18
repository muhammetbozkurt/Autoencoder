# Autoencoder
Before started I did not share dataset because I am not sure if I am allowed to share it. Because I got dataset from a challenge. 

I planed this repo as first part of a series. In this paper, firstly “What is Autoencoders” was answered then I explained how I build a CNN based autoencoder for signature images and showed its results.

## What is Autoencoder?

  An autoencoder is an unsupervised machine learning algorithm which aims to reconstruct input. It may look a little bit odd when reading it first time but algorithm does not directly copy input layer to output layer. As figure 1 represent, algorithm first reduces representation then reconstructs input from reduced representation. This process provides capturing useful properties in input data to algorithm. A noteworthy feature of autoencoders is while getting compressed version of input data, noise in input data is removed.

  First part of model which is input layer to middle layer is named as encoder part and other part of algorithm which is middle layer to output layer is named  as decoder part. Encoded features of input data is obtained from middle layer that is also referred as bottleneck.

![Figure 1](https://github.com/muhammetbozkurt/Autoencoder/blob/master/autoencoder_rep.png)
###### *Figure 1: A basic autoencoder with a single layer (Charu C. Aggarwal, Neural Networks and Deep Learning, p.73)*


## Building Our CNN Autoencoder model

  When building model I first directly used VGG-16 architecture as encoder part (therefore reversing it for decoder part). However, it had 31,470,529 trainable parameters which does not provide efficiency in training phase and also to train such a network need huge amount of data* so I chanced it a bit. You can check altered architecture at the end of the page (figure 2). 
  
  In last version of model have 308,593  trainable parameters which looks feasible. Moreover, model is so deep which means gradient vanishing problem can be occurred. To prevent gradient vanishing problem, batch normalization layers were added and also relu activation function were used in hidden layers. 

  A data generator function is coded for generating noisy unique data in training phase for every batch which described in details below. Every epoch 32 * 10000 data point will be generated.

  After preprocess (detailed below), we have a binary structure in our data so sigmoid is output layer activation function and binary cross entropy is our loss function.

Note: This model trained on Google Colab

**rule of tumb is the total number of training data points should be at least 2 or 3 times larger than number of parameters in neural networks*

## Preporcess

Sign images in our dataset only contains 0 or 255 as their pixels  values and every image has  different resolutions.

1. Resize image to (224, 224) using cv2.INTER_NEAREST as interpolation technique to preserve the binary structure in images.

2. Divide every pixels to 255 and Convert them float type.

After these steps pixels values are 1 or 0.

## Data Generation

data_generator funtion first reads images from training dataset then flips and rotates images (or not) at the end applies salt and pepper method to input image to add noise. As we mentioned the main purposes of autoencoders are removing noise in training data and dimentionality reduction. Therefore adding noise is important for our autoencoder to gain ability to remove noise.  

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
