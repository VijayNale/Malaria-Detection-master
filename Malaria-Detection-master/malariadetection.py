# -*- coding: utf-8 -*-
"""
@author: Krish.Naik
"""

## MAlaria Detection using Transfer Learning

# Download The Dataset from 
#https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria


# dataset has blood test 
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
#from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'cell_images/Train'
valid_path = 'cell_images/Test'


# add preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)  #3 means RGB(red green black) image 

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  

  # useful for getting number of classes
folders = glob('cell_images/Train/*')
  

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)  #softmax is activation function to same as sigmoid , to predict bet 0 to 1

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# Use the Image Data Generator to import the images from the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('cell_images/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('cell_images/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# fit the model : 1 hr time to build
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

"""
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Epoch 1/5
32/32 [==============================] - 449s 14s/step - loss: 0.5947 - acc: 0.7313 - val_loss: 0.6260 - val_acc: 0.7200
Epoch 2/5
32/32 [==============================] - 1075s 34s/step - loss: 0.2687 - acc: 0.8945 - val_loss: 0.3286 - val_acc: 0.8550
Epoch 3/5
32/32 [==============================] - 438s 14s/step - loss: 0.2067 - acc: 0.9219 - val_loss: 0.3873 - val_acc: 0.8300
Epoch 4/5
32/32 [==============================] - 1649s 52s/step - loss: 0.1793 - acc: 0.9335 - val_loss: 0.7075 - val_acc: 0.7300
Epoch 5/5
32/32 [==============================] - 446s 14s/step - loss: 0.2049 - acc: 0.9230 - val_loss: 0.5218 - val_acc: 0.7900
"""

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model
model.save('model_vgg19.h5')


from keras.preprocessing import image
img_path = 'malaria_image.png'
img = image.load_img(img_path, target_size=(224, 224))
img
x = image.img_to_array(img)
x
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

parasitised = model.predict(x)
parasitised  #=> array([[1., 0.]], dtype=float32)

model.layers
