import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2,os
from keras import models

print("tensorflow version",tf.__version__)

def my_preprocessing_func(img):
    image = np.array(img)
    return image / 255
print('getcwd:      ', os.getcwd())
filename=(r"src\features\PotatoEarlyBlight1.JPG")
plant_color = cv2.imread(filename, cv2.IMREAD_COLOR)
print('le type de img_color est :', type(plant_color))
plant_colorv2 = cv2.cvtColor(plant_color, cv2.COLOR_BGR2RGB)
resize_down = cv2.resize(plant_colorv2, (256,256), interpolation= cv2.INTER_LINEAR)
np_img=my_preprocessing_func(resize_down)
print("shape",np_img.shape)
d=np_img.reshape(1,256,256,3)
print("reshape",d.shape)

#restoration model
cnn=models.load_model(r"D:\\Adrien et Sarah\\Documents\\GitHub\\models\\cnn4.keras")
pred=cnn.predict(d) # type: ignore

predn=pred.argmax(axis=1)[0]

labels=pd.read_csv("references/labels.csv");

print(labels.iloc[predn].values)