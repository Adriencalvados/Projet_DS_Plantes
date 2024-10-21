# Import des librairies
# Base

import numpy as np # type: ignore
import pandas as pd # type: ignore
import time,os
import cv2# type: ignore

# Data Viz
import matplotlib.pyplot as plt# type: ignore
import seaborn as sns# type: ignore

# Modèle
import tensorflow as tf# type: ignore
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D,Input,Conv2D,MaxPooling2D,MaxPool2D# type: ignore
from keras import layers, models, optimizers, losses, metrics,callbacks,utils,Model,Sequential,Input# type: ignore

# Rééchantillonnage 
from sklearn.model_selection import train_test_split# type: ignore

# Evaluation et métriques
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay# type: ignore

# Transformation
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
print("tensorflow version",tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




def my_preprocessing_func(img):
    image = np.array(img)
    return image / 255

print('getcwd:      ', os.getcwd())
#selection only_healthy
if 1==1:
    directory1=(r"./JAN24_PLANT_RECOGNITION/New_Plant_Diseases_Dataset_only_healthy/train")
    directory2=(r"./JAN24_PLANT_RECOGNITION/New_Plant_Diseases_Dataset_only_healthy/valid")
    directory3=(r"./JAN24_PLANT_RECOGNITION/New_Plant_Diseases_Dataset_only_healthy/test")
#selection all
if 0==1:
    directory1=(r"./JAN24_PLANT_RECOGNITION/01_New_Plant_Diseases_Dataset/train")
    directory2=(r"./JAN24_PLANT_RECOGNITION/01_New_Plant_Diseases_Dataset/valid")
    directory3=(r"./JAN24_PLANT_RECOGNITION/01_New_Plant_Diseases_Dataset/test")   
print('le filename de img_color est :', directory1)

#augmentation de donnée : rotation_range=10, width_shift_range=0.1,  height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,
train_gen = ImageDataGenerator(preprocessing_function=my_preprocessing_func)
val_gen = ImageDataGenerator(preprocessing_function=my_preprocessing_func)   
test_datagen = ImageDataGenerator(preprocessing_function=my_preprocessing_func)

train_generator=train_gen.flow_from_directory(
    directory=directory1,
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=32,
    shuffle=False,
    class_mode="categorical"
)

valid_generator = val_gen.flow_from_directory(
    directory=directory2,
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=directory3,
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=42
)


cnn = Sequential()
#cnn.add(Input(shape=(256,256,3),dtype=tf.float32))
cnn.add(layers.Conv2D(32,(3, 3),padding='same',activation='relu',input_shape=(256,256,3)))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(256, (3, 3), activation='relu'))
cnn.add(layers.Dropout(0.25))
cnn.add(layers.Flatten())
cnn.add(layers.Dense(units=512,activation='relu'))#1500
cnn.add(layers.Dropout(0.4)) #To avoid overfitting
#Output Layer
cnn.add(layers.Dense(units=len(train_generator.class_indices),activation='softmax'))
cnn.compile(optimizer=optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
cnn.summary()


early_stopping = callbacks.EarlyStopping(monitor = 'val_loss',
                                        patience = 10,
                                        mode = 'min',
                                        restore_best_weights = True)

lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                        patience=5,
                                        factor=0.5,
                                        verbose=2,
                                        mode='min',
                                        min_lr = 1e-10) # type: ignore

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
#Model.fit_generator est obsolète depuis TensorFlow 2.1.0
#vous pouvez le corriger en utilisant Model.fit pour utiliser ImageDataGenerator comme entrée

# entrainement avec la méthode .fit()
history = cnn.fit(train_generator, 
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=2,#9
                    callbacks=[early_stopping,lr_plateau]
)
cnn.save(r"./JAN24_PLANT_RECOGNITION/cnn1.keras")


import pickle

# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([history], f)

# Getting back the objects:
#with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
#    obj0, obj1, obj2 = pickle.load(f)