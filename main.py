import urllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import time
import urllib3
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,Input,Dropout
from keras.models import Model,Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras.optimizers

data = pd.read_json('Indian_Number_plates.json',lines=True)
print(data.head(4))
dataset = dict()
dataset['image_name'] = list()
dataset['image_width'] = list()
dataset['image_height'] = list()
dataset['top_x'] = list()
dataset['top_y'] = list()
dataset['bottom_x'] = list()
dataset['bottom_y'] = list()
print(dataset)
counter = 0
for index,row in data.iterrows():
    try:
        img = urllib.request.urlopen(row["content"])
        img = Image.open(img)
        img = img.convert('RGB')
        img.save('Indian Number Plates/licensed car {}'.format(counter))
        dataset['image_name'].append("licensed_car{}".format(counter))
        data = row["annotation"]
        dataset["image_width"].append(data[0]["image_width"])
        dataset["image_height"].append(data[0]["image_height"])
        dataset["top_x"].append(data[0]["points"][0]["x"])
        dataset["top_y"].append(data[0]["points"][0]["y"])
        dataset["bottom_x"].append(data[0]["points"][1]["x"])
        dataset["bottom_y"].append(data[0]["points"][1]["y"])
        counter = counter + 1
    except:
        pass

dataFrame = pd.DataFrame(dataset)
dataFrame.to_csv("Indian License Plate.csv",index=False)
dataFrame = pd.read_csv('Indian_Number_plates.json')
print(dataFrame.head(4))