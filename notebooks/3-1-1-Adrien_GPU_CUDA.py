# Import des librairies
# Base
from mlflow import MlflowClient
import mlflow
import mlflow.keras
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
from tensorflow.keras.applications import VGG19
# Rééchantillonnage 
from sklearn.model_selection import train_test_split# type: ignore

# Evaluation et métriques
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay# type: ignore

# Transformation
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
print("tensorflow version",tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define tracking_uri
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
cnn_experiment = mlflow.set_experiment("Plants_Models")
run_name = "run_15_Plant_Diseases_Dataset_off_outliers"
#mettre disable true pour ne pas avoir les artefacts
mlflow.tensorflow.autolog()

#filtre defaut lors du fit

def my_preprocessing_func(img):
    image = np.array(img)
    return image / 255

print('getcwd:      ', os.getcwd())
#selection only_healthy
if 0==1:
    directory1=(r"D:\Adrien et Sarah\Documents\GitHub\models\DATASETS\Maladie_sur_les_plantes\New_Plant_Diseases_Dataset_only_healthy\train")
    directory2=(r"D:\Adrien et Sarah\Documents\GitHub\models\DATASETS\Maladie_sur_les_plantes\New_Plant_Diseases_Dataset_only_healthy\valid")
#selection all
if 0==1:
    directory1=(r"D:\Adrien et Sarah\Documents\GitHub\models\DATASETS\Maladie_sur_les_plantes\01_New_Plant_Diseases_Dataset\train")
    directory2=(r"D:\Adrien et Sarah\Documents\GitHub\models\DATASETS\Maladie_sur_les_plantes\01_New_Plant_Diseases_Dataset\valid")
#03_Plant_Diseases_Dataset_off_outliers
if 1==1:
    directory1=(r"../JAN24_PLANT_RECOGNITION/03_Plant_Diseases_Dataset_off_outliers/train")
    directory2=(r"../JAN24_PLANT_RECOGNITION/03_Plant_Diseases_Dataset_off_outliers/valid")
#04_04_tomates_malades
if 0==1:
    directory1=(r"../JAN24_PLANT_RECOGNITION/04_tomates_malades/train")
    directory2=(r"../JAN24_PLANT_RECOGNITION/04_tomates_malades/valid")

print('le filename de img_color est :', directory1)

#augmentation de donnée : rotation_range=10, width_shift_range=0.1,  height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,
train_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,  height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,preprocessing_function=my_preprocessing_func)
val_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,  height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,preprocessing_function=my_preprocessing_func)   

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

from tensorflow.keras.optimizers.legacy import Adam

with mlflow.start_run(
    run_name=run_name,
    nested=False,
    tags={"project": "image_classification", "model": "transfert_learning-VGG19-finetuning"}
                    ):
    # Modèle VGG19
    base_model = VGG19(include_top=False,
    weights="imagenet",
    #input_tensor=None,
    input_shape=(256,256,3),
    #pooling=None,
    #classifier_activation="softmax")
    )
# Freezer les couches
    for layer in base_model.layers:
        layer.trainable = False
    # Déverrouiller les  dernières couches
    for layer in base_model.layers[-8:]:
        layer.trainable = True
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(2048, activation='relu')(x)
    x= layers.Dropout(rate=0.2)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x= layers.Dropout(rate=0.2)(x)
    output = layers.Dense(38, activation='softmax')(x)
    cnn = models.Model(inputs=[base_model.input], outputs=[output])
    
    cnn.load_weights("mlruns/902956440979823799/863934c429124c9ea23058ba839f7ec2/artifacts/model/data/model.keras")
    
    cnn.compile(optimizer=optimizers.Adam(learning_rate=1e-8),loss='categorical_crossentropy',metrics=['accuracy'])
    cnn.summary()

    early_stopping = callbacks.EarlyStopping(monitor = 'val_accuracy',
                                            patience = 4,
                                            mode = 'auto',
                                            restore_best_weights = True,
                                            verbose=1,
                                            start_from_epoch=5
                                            )

    lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',
                                            patience=4,
                                            factor=0.1,
                                            verbose=1,
                                            mode='auto',
                                            min_lr = 1e-10,
                                            cooldown=2) # type: ignore

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    #Model.fit_generator est obsolète depuis TensorFlow 2.1.0
    #vous pouvez le corriger en utilisant Model.fit pour utiliser ImageDataGenerator comme entrée
    # entrainement avec la méthode .fit()
    cnn.fit(train_generator, 
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,#22
                    callbacks=[early_stopping,lr_plateau,[mlflow.tensorflow.MlflowCallback()]]
            )

