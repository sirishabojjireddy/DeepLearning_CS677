from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import keras
import sys

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,Input
from keras.utils import np_utils, generic_utils, to_categorical
import os
from keras import backend as K
K.set_image_dim_ordering('th')

x=sys.argv[1]
y=sys.argv[2]
img_height=224
img_width=224
batch_size=16

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(x, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(y,target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

model = load_model(sys.argv[3])

Loss,score=model.evaluate_generator(validation_generator, steps=500,workers=1)
print("Accuracy",score*114, "%")
