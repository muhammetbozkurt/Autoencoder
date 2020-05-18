#py version
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

import matplotlib.pyplot as plt

import cv2
import os
import random
import numpy as np
from tqdm import tqdm

SHAPE = (224,224,1)


#Batch Normalization is usually inserted after fully connected layers or Convolutional layers and before non-linearity
def create_model(orginal =False, input_shape = (224, 224, 1)):
    #inspired from vgg16
    if(orginal):
        input_img = Input(shape=input_shape) 

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2, 2), padding='same')(x)#112 112 64
        
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2, 2), padding='same')(x)#56 56 128

        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2, 2), padding='same')(x)#28 28 128

        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((2, 2), padding='same')(x)#14 14 512

        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        encoded = MaxPooling2D((2, 2), padding='same', name="encoder")(x)#7 7 512
        
        x = UpSampling2D((2, 2))(encoded)#14 14 512
        
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D((2, 2))(x)#28 28 512

        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D((2, 2))(x)#56 56 512

        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D((2, 2))(x)#112 112 256

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D((2, 2))(x)#224 224 128

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        return Model(input_img, decoded)
    input_img = Input(shape=input_shape) 

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)#112 112 16
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)#56 56 32

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2), padding='same')(x)#28 28 32
    #-------
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)#14 14 64

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    encoded = MaxPooling2D((2, 2), padding='same', name="encoder")(x)#7 7 64
    
    x = UpSampling2D((2, 2))(encoded)#14 14 64
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)#28 28 64

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    #-------
    x = UpSampling2D((2, 2))(x)#56 56 64

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)#112 112 64

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)#224 224 32

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(input_img, decoded)


def pepper(image):
    amount = 0.2
    out = image.copy()

    # Pepper mode
    num_pepper = np.ceil(amount* image.size )
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out

def flip(image):
    if(random.randint(0,1) == 1):
        return cv2.flip(image,1) 
    return image

def rotate(image):
    prob = random.randint(0,3)
    if(prob == 0):
        return cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif(prob == 1):
        return cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    elif(prob == 2):
        return cv2.rotate(image,cv2.ROTATE_180)
    return image


def data_generator(directory, batch_size):

    i = 0
    file_list = os.listdir(directory)
    random.shuffle(file_list)
    while True:
        train_x = []
        train_y = []
        for _ in range(batch_size):
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)
            sample = file_list[i]
            i += 1
            
            image = cv2.resize(cv2.imread("{}/{}".format(directory,sample),0), (SHAPE[0], SHAPE[1]), interpolation=cv2.INTER_NEAREST) # INTER_NEAREST to protect binary structure
    
            image = flip(image)
            image = rotate(image)
            image = image.astype(float)/255.
            #if random.randint(0,1)<1:
            #    image = pepper(image)
            train_x.append(pepper(image))
            train_y.append(image)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        #print("---------------------------")
        yield train_x.reshape((-1, SHAPE[0], SHAPE[1], 1)), train_y.reshape((-1, SHAPE[0], SHAPE[1], 1))
        #yield train_y.reshape((-1, SHAPE[0], SHAPE[1], 1))

def compile_model(model):
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.99, nesterov=True)
    # model.compile(optimizer='nadam', loss=custom_loss)
    model.compile(optimizer=sgd, loss="binary_crossentropy")
    return model

def train_with_generator_model(model, directory):

    model = compile_model(model)
    
    checkpoint = ModelCheckpoint("/content/drive/My Drive/challange_data/bin_v{epoch:02d}.hdf5", verbose=1) #val_acc yok ondan val_loss
    
    """
    fit_generator(object, generator, steps_per_epoch, epochs = 1,
        verbose = getOption("keras.fit_verbose", default = 1),
        callbacks = NULL, view_metrics = getOption("keras.view_metrics",
        default = "auto"), validation_data = NULL, validation_steps = NULL,
        class_weight = NULL, max_queue_size = 10, workers = 1,
        initial_epoch = 0)
    """
    
    model.fit_generator(data_generator(directory, 32),
                        steps_per_epoch = 10000,
                        epochs = 10,
                        validation_data=data_generator(directory, 100),
                        validation_steps = 5,
                        callbacks=[checkpoint],
                        verbose=1)
    model.save("/content/drive/My Drive/challange_data/binary.hdf5")


def train_model(model, batch_size = 50, dataset_size = 1000):

    model = compile_model(model)
    
    checkpoint = ModelCheckpoint("/content/drive/My Drive/challange_data/bin_v{epoch:02d}.hdf5", monitor='vall_acc', verbose=1) #val_acc yok ondan val_loss
    
    a = data_generator("/content/drive/My Drive/challange_data/train_data", dataset_size)
    
    x = next(a)
    print(x.shape)
    #print(y.shape)
    print("batch_size: ",batch_size)
    model.fit(x, x,
                        epochs=10,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split = 0.1,
                        callbacks=[checkpoint ])
    model.save("/content/drive/My Drive/challange_data/binary.hdf5")



def retrain(path_to_model, directory, epoch = 5):

    model = load_model(path_to_model)
    compile_model(model)

    checkpoint = ModelCheckpoint("/content/drive/My Drive/challange_data/re_bin_v{epoch:02d}.hdf5", verbose=1) #val_acc yok ondan val_loss

    
    model.fit_generator(data_generator(directory, 32),
                        steps_per_epoch = 10000,
                        epochs = epoch,
                        validation_data=data_generator(directory, 100),
                        validation_steps = 5,
                        callbacks=[checkpoint],
                        verbose=1)
    model.save("/content/drive/My Drive/challange_data/re_binary.hdf5")
    
def sample_output_generate(model, num_of_sample = 5, directory="/content/drive/My Drive/challange_data/test_data"):
    dir_list = os.listdir(directory)
    for i in range(1,2*num_of_sample+1,2):
        file_name = dir_list[random.randint(0,len(dir_list))]
        image = cv2.resize(cv2.imread("{}/{}".format(directory, file_name),0), (SHAPE[0], SHAPE[1]), interpolation=cv2.INTER_NEAREST)
        
        copy = image.copy()

        image = image.astype(float)/255.
        res = re_auto.predict(image.reshape((1, 224,224,1)))
        res = res.reshape((224, 224))*255
        res = np.array(res, dtype=np.uint8)
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(copy,cmap="gray")
        ax2.imshow(res,cmap="gray")
