#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.layers import Dense, Dropout, Input, UpSampling2D, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf
import keras
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, Flatten, Concatenate
from tensorflow.keras.activations import relu
from segmentation_models import PSPNet, Unet
import segmentation_models as sm
#from patchify import patchify, unpatchify
import albumentations as A
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import cv2
from tensorflow.keras.models import load_model

# In[31]:


sm.framework()


# In[32]:


sm.set_framework('tf.keras')


# In[33]:


class StitchPatches(tf.keras.layers.Layer):
    def __init__(self , batch_size):
        super(StitchPatches , self).__init__()
        self.batch_size = batch_size

    def call(self, inputs):
        print(inputs)
        patches = []
        main_image = np.empty([inputs.shape[0], 192,192,inputs.shape[3]])
        for k in range(0, inputs.shape[0], self.batch_size):
            for i in range(0 ,192, 48):
                for j in range(0 ,192 , 48):
                    main_image[i : i + 48 , j : j + 48 , : ] = inputs[k]
        return main_image


# In[34]:


class CreatePatches( tf.keras.layers.Layer ):

  def __init__( self , patch_size ):
    super( CreatePatches , self ).__init__()
    self.patch_size = patch_size

  def call(self, inputs ):
    patches = []
    # For square images only ( as inputs.shape[ 1 ] = inputs.shape[ 2 ] )
    input_image_size = inputs.shape[ 1 ]
    for i in range( 0 , input_image_size , self.patch_size ):
        for j in range( 0 , input_image_size , self.patch_size ):
            patches.append( inputs[ : , i : i + self.patch_size , j : j + self.patch_size , : ] )
    return patches

#sample_image = np.random.rand(1, 192 , 192 , 3 ).astype(np.float32)
in1 = tf.keras.Input(shape=( 192,192,3))
in2 = tf.keras.Input(shape=( 192,192,3))
#input = (Input(shape=(192, 192, 3), name='input'))
layer = CreatePatches( 48 )
#print(layer)
layer = layer(in1)


# In[35]:
type_model_class4 = load_model('./mobilenet_4classes_model.h5')

local_model = PSPNet(backbone_name='mobilenet',input_shape=(48, 48, 3),classes=1,activation='sigmoid', encoder_freeze=False)
for i in np.arange(37):
   local_model.layers[i].set_weights(type_model_class4.layers[i].get_weights())

# In[36]:


out0 = local_model(layer[0])
out1 = local_model(layer[1])
out2 = local_model(layer[2])
out3 = local_model(layer[3])
out4 = local_model(layer[4])
out5 = local_model(layer[5])
out6 = local_model(layer[6])
out7 = local_model(layer[7])
out8 = local_model(layer[8])
out9 = local_model(layer[9])
out10 = local_model(layer[10])
out11 = local_model(layer[11])
out12 = local_model(layer[12])
out13 = local_model(layer[13])
out14 = local_model(layer[14])
out15 = local_model(layer[15])


# In[37]:


def putall(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = x
    return K.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16], axis=1)


# In[38]:


X_patch = Lambda(putall)([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11,
                        out12, out13, out14, out15])
print(X_patch)


# In[39]:


#out_combined = tf.stack([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15], axis=1)


# In[40]:


def merge_patches(x):
    return K.reshape(x,(-1,192, 192, 1))


# In[41]:


#rec_new = tf.space_to_depth(X_patch[-1],4)
#rec_new = tf.reshape(rec_new,[-1,192,192,1])


# In[42]:


X_patch = Lambda(merge_patches)(X_patch)


# In[43]:


#global_model = PSPNet(backbone_name='densenet121',input_shape=(192, 192, 3),classes=1,activation='sigmoid', encoder_freeze=False)

#for i in np.arange(140):
#    global_model.layers[i].set_weights(type_model_class4.layers[i].get_weights())
# In[44]:


#X_global_output = global_model(in2)


# In[45]:


#X_final = Concatenate(axis=3)([X_patch, X_global_output])
X_final = Conv2D(1, 1, activation='sigmoid')(X_patch)


# In[46]:


model_1 = Model(inputs=in1, outputs=X_final)
model_1.summary()


# In[47]:


#data_dir = "./sample_corrected/sample/"
data_dir = "./train/sample/"
#data_dir = "./miccai2/images/"
#mask_dir = "./sample_corrected/mask/"
mask_dir = "./train/masked_images/"
#mask_dir = "./miccai2/labels/"


# In[13]:


import os
from sklearn.model_selection import train_test_split
all_images = os.listdir(data_dir)

to_train = 1  # ratio of number of train set images to use
total_train_images = all_images[:int(len(all_images) * to_train)]
len(total_train_images)


# In[14]:


# split train set and test set
train_images, validation_images = train_test_split(total_train_images, train_size=0.8, test_size=0.2, random_state=0)
print(len(train_images), len(validation_images))


# In[50]:


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        #A.ShiftScaleRotate(p=0.5),
        A.RandomRotate90(),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.Blur(blur_limit=3),
        A.RandomBrightnessContrast(p=0.2),
        #A.RGBShift(p=0.2),
        #A.HueSaturationValue(),
        A.Transpose(),
    ],
    additional_targets={'image0': 'image'}
)


# In[56]:


BATCH_SIZE = 16
width = 192
height = 192


# In[65]:


def normalize(arr):
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    return arr


# In[75]:


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
            image_batch1 = []
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
                    transformed = transform(image=image, image0=label)
                    aug_img = transformed['image']
                    aug_mask = transformed['image0']
                    #print(label.shape)
                    image_batch.append(aug_img)
                    #image_batch1.append(aug_img)
                    label_batch.append(aug_mask)
                i += 1
            if image_batch and label_batch:
                image_batch = normalize(np.array(image_batch))
                #image_batch1 = normalize(np.array(image_batch1))
                label_batch = normalize(np.array(label_batch))
                yield ([image_batch], label_batch)


# In[76]:


train_gen = generate_data(train_images, BATCH_SIZE, (width, height), train=True )
val_gen = generate_data(validation_images, BATCH_SIZE, (width, height), val=True)


# In[61]:


epochs = 100

callbacks = [
    ModelCheckpoint("./wacv/pspnet_wstech_imagenet1_nofreeze_mobilenet.h5", save_weights_only=True, save_best_only=True, mode='min')
]
model_1.compile(
    'Adam',
    loss=sm.losses.DiceLoss(),
    metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), 'binary_accuracy'],
)


# In[ ]:


model_1.fit(train_gen, steps_per_epoch=np.ceil(float(len(train_images)) / float(BATCH_SIZE)), epochs=epochs, callbacks=callbacks, validation_data=val_gen, validation_steps=np.ceil(float(len(validation_images)) / float(BATCH_SIZE)), verbose=1)


# In[ ]:




