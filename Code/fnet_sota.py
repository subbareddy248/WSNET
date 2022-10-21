#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow_addons as tfa
import PIL
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
from tensorflow.keras import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import os
from tensorflow.keras.backend import flatten
import tensorflow.keras.backend as K
from tensorflow.keras import layers

#env SM_FRAMEWORK=tf.keras
import  segmentation_models as sm
from segmentation_models import Unet, Linknet, PSPNet, FPN
from segmentation_models.utils import set_trainable
import cv2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model

# In[2]:


data_dir = "./train/sample/"
#data_dir = "./miccai2/images/"
#data_dir = "./medetec/images/"
mask_dir = "./train/masked_images/"
#mask_dir = "./miccai2/labels/"
#mask_dir = "./medetec/labels/"

all_images = os.listdir(data_dir)

to_train = 1  # ratio of number of train set images to use
total_train_images = all_images[:int(len(all_images) * to_train)]

WIDTH = 224  # actual : 1918//1920 divisive by 64
HEIGHT = 224  # actual : 1280
BATCH_SIZE = 16


# In[6]:


# In[52]:


def normalize(arr):
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    return arr
def generate_data(images_list, batch_size, dims, train=False, val=False, test=False):
        """Replaces Keras' native ImageDataGenerator."""
        try:
            if train is True:
                #print(images_list)
                image_file_list = images_list
                label_file_list = images_list
            elif val is True:
                image_file_list = images_list
                label_file_list = images_list
            elif test is True:
                image_file_list = images_list
                label_file_list = images_list
        except ValueError:
            print('one of train or val or test need to be True')

        i = 0
        while True:
            image_batch = []
            label_batch = []
            for b in range(batch_size):
                if i == len(image_file_list):
                    i = 0
                if i < len(image_file_list):
                    sample_image_filename = image_file_list[i]
                    sample_label_filename = label_file_list[i]
                    #print('image: ', image_file_list[i])
                    #print('label: ', label_file_list[i])
                    if train or val:
                        image = cv2.imread(data_dir + sample_image_filename, 1)
                        image = cv2.resize(image,dims)
                        label = cv2.imread(mask_dir + sample_label_filename, 0)
                        label = cv2.resize(label, dims)
                        #print(label.shape)
                    elif test is True:
                        image = cv2.imread(data_dir + sample_image_filename, 1)
                        image = cv2.resize(image, dims)
                        label = cv2.imread(mask_dir + sample_label_filename, 0)
                        label = label.resize(label,dims)
                    # image, label = self.change_color_space(image, label, self.color_space)
                    label = np.expand_dims(label, axis=2)
                    #print(label.shape)
                    image_batch.append(image)
                    label_batch.append(label)
                i += 1
            if image_batch and label_batch:
                image_batch = normalize(np.array(image_batch))
                label_batch = normalize(np.array(label_batch))
                yield (image_batch, label_batch)

# Now let's use Tensorflow to write our own dice_coeficcient metric, which is a effective indicator of how much two sets overlap with each other
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_p_bce(in_gt, in_pred):
    return 0.0*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


backbones = ['densenet121','densenet169','resnet50','seresnet50','seresnext50']
backbones = ['mobilenet']
df_metrics = pd.DataFrame(columns=['architecture', 'backbone', 'train_ratio','cross_fold','loss','binary_accuracy','dice_coef','IOUScore','FScore'])
train_ratio = [0.8, 0.7, 0.6, 0.5, 0.4]
type_model_class4 = load_model('./mobilenet_4classes_model.h5')
layer_dict = dict([(layer.name, layer) for layer in type_model_class4.layers])

# add one more for loop to add architectures and append 'architecture to df_metrics..
for eachbackbone in backbones:
    for split_ratio in  train_ratio:
        for fold in np.arange(1):
            # split train set and test set
            train_images, validation_images = train_test_split(total_train_images, train_size=split_ratio, test_size=0.2, random_state=7)
            #validation_images = validation_images[:200]
            # generator for train and validation data set
            train_gen = generate_data(train_images, BATCH_SIZE, (WIDTH, HEIGHT), train=True )
            val_gen = generate_data(validation_images, BATCH_SIZE, (WIDTH, HEIGHT), val=True)

            model = FPN(
                backbone_name=eachbackbone,
                encoder_freeze=False,
                classes=1,
                activation='sigmoid'
            )
            for i in np.arange(len(type_model_class4.layers)-3):
                model.layers[i].set_weights(type_model_class4.layers[i].get_weights())

            # In[9]:
            dice_loss = sm.losses.DiceLoss()
            focal_loss = sm.losses.BinaryFocalLoss()
            total_loss = dice_loss + (1 * focal_loss)
            total_loss = sm.losses.binary_focal_dice_loss

            callbacks = [
                ModelCheckpoint('./fpn_models/best_model_fpnet_'+str(eachbackbone)+'_'+str(fold)+'_'+str(split_ratio)+'.h5', save_weights_only=True, save_best_only=True, mode='min'),
                ReduceLROnPlateau(),
            ]

            model.compile('Adam', dice_loss, ['binary_accuracy', dice_coef, sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)])
            history = model.fit(train_gen,steps_per_epoch=np.ceil(float(len(train_images)) / float(BATCH_SIZE)),
                                epochs=100,
                                validation_steps=np.ceil(float(len(validation_images)) / float(BATCH_SIZE)),
                                validation_data = val_gen,
                                callbacks = callbacks)
            np.save('./fpn_models/fpnet_history_'+str(eachbackbone)+'_'+str(fold)+'_'+str(split_ratio)+'.npy',history.history)
            val_gen = generate_data(validation_images, BATCH_SIZE, (WIDTH, HEIGHT), val=True)
            results = model.evaluate_generator(val_gen, steps=np.ceil(float(len(validation_images)) / float(BATCH_SIZE)))

            df_metrics = df_metrics.append({'architecture':'Linknet', 'backbone':eachbackbone, 'train_ratio':split_ratio, 'cross_fold':fold,'loss':results[0],
                            'binary_accuracy':results[1],'dice_coef':results[2],'IOUScore':results[3],'FScore':results[4]}, ignore_index=True)
        break      
    df_metrics.to_csv('./fpn_models/fpnet_history_'+str(eachbackbone)+'_'+str(split_ratio)+'.npy')

