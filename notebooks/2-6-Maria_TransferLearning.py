# -*- coding: utf-8 -*-

import tensorflow as tf
print("tensorflow version",tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np

import matplotlib.pyplot as plt
# import time, cv2
import seaborn as sns


import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

where= "/content/drive/MyDrive/ColabNotebooks/Dataset/Etape 3 - Maladie sur les plantes sans outlier/03_Plant_Diseases_Dataset_off_outliers"
train= where+'/train'
valid= where+'/valid'
test= where+'/test'

train_data_generator = ImageDataGenerator(rescale=1.0/255.0)

valid_data_generator = ImageDataGenerator(rescale=1.0/255.0)

test_data_generator = ImageDataGenerator(rescale=1.0/255.0)

batch_size = 32


train_generator = train_data_generator.flow_from_directory(directory=train,
                                                           target_size=(256, 256),color_mode="rgb",batch_size=batch_size,shuffle=False,class_mode="categorical")

valid_generator = valid_data_generator.flow_from_directory(directory=valid,
                                                           target_size=(256, 256),color_mode="rgb",batch_size=batch_size,shuffle=False,class_mode="categorical")

test_generator = test_data_generator.flow_from_directory(directory=test,
                                                         target_size=(256, 256),color_mode="rgb",batch_size=1,shuffle=False,class_mode="categorical")

from tensorflow.keras.applications import ResNet152V2

n_class=38
# Modèle ResNet152V2
base_model = ResNet152V2(    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(256,256,3),
    pooling=None,
    classifier_activation="softmax")
# Freezer les couches
for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model) # Ajout du modèle base
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(n_class, activation='softmax'))
# model.add(Flatten())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()

# tailes des échantillons
nb_img_train = train_generator.samples
nb_img_test = test_generator.samples

# entrainement avec la méthode .fit()
history =model.fit(train_generator, epochs=5, validation_data=valid_generator, steps_per_epoch=nb_img_train // 32, validation_steps=nb_img_test // 32)

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss by epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.subplot(122)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model acc by epoch')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')
plt.show()

for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
history =model.fit(train_generator, epochs=5, validation_data=valid_generator, steps_per_epoch=nb_img_train // 32, validation_steps=nb_img_test // 32)

plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss by epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.subplot(122)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model acc by epoch')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')
plt.show()

model.save("/content/drive/MyDrive/ColabNotebooks/transfer_learning_model.h5")