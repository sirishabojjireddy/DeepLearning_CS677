import sys
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense,Input
from keras.utils import np_utils, generic_utils, to_categorical
from keras.metrics import categorical_accuracy
import os
from keras import backend as K
K.set_image_dim_ordering('th')

img_width, img_height = 224,224
input_tensor = Input(shape=(3,224,224))

train_data_dir =sys.argv[1] 
validation_data_dir = sys.argv[2]

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 20
batch_size = 16
inout_shape=[224,224,3]

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
print('Model loaded.....')

for layer in model.layers[:]:
    layer.trainable=False

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dropout(0.5))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(4096, activation='relu', name='fc2'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='sigmoid'))

model1 = Model(inputs = model.input, outputs =top_model(model.output))
model1.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.1), metrics=['categorical_accuracy'])

# fine-tune the model
model1.fit_generator(train_generator, samples_per_epoch=nb_train_samples, epochs=epochs, validation_data=validation_generator,nb_;ansamples=nb_validation_sampl

model1.save(sys.argv[3])

