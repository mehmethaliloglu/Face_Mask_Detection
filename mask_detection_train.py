# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:06:15 2020

@author: monster
"""

# Libraries

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense,Input,Flatten,Dropout,AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os 
#%%

# I kept the names of the pictures with and without masks in the list.

path="C:/Users/monster/Desktop/mask_detection_project/"
masked=[]
unmasked=[]

for name in os.listdir(path+"Mask"):
    masked.append(name)
for name in os.listdir(path+"No_Mask"):
    unmasked.append(name)
    
masked_and_unmasked=masked+unmasked

#%%

# I read the pictures as per the list above. I set its dimensions to 210x210. 
# I converted these pictures to array and kept them at X.
# In Y, I kept the value 0 if the Images are masked, and the value 1 if the images are not masked.

x=[]
y=[]
for name in masked_and_unmasked:
    if name in masked:
        label=0
        image=load_img(path+"Mask/"+str(name))
    else:
        label=1
        image=load_img(path+"No_Mask/"+str(name))
    image=img_to_array(image)
    image=preprocess_input(image)
    image=cv2.resize(image,dsize=(210,210),interpolation=cv2.INTER_CUBIC)
    x.append(image)
    y.append(label)
x=np.array(x)
y=np.array(y)
#%%

# LabelBinarizer was used to show Y in columns 1-0. It could be used in OneHotEncoder.

labelbinarizer=LabelBinarizer()
y=labelbinarizer.fit_transform(y)
y=to_categorical(y)

#%%

# Train-Test distinction was made for the model.

X_train,X_test,Y_train,Y_test=train_test_split(x, y,test_size=0.2,random_state=2020)

#%%

# I have 5000 images and used ImageDataGenerator to work on different variations of the same images.
# This means the image will increase the number and see different variations.

image_data_generator=ImageDataGenerator(rotation_range=30,zoom_range=0.15,width_shift_range=0.2,
                                        height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

#%%

# As a model, I stuck between InceptionResNetV2 and MobileNetV2 and tried both.
# I used it because MobileNetV2 works better. I gave input_tensor the image sizes I set.

mobile_net_v2 = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(210, 210, 3)))

#%%

# Before creating the model, we defined the middleware values ​​and activation values ​​for the model.

before_model=mobile_net_v2.output
before_model=AveragePooling2D(pool_size=(2,2))(before_model)
before_model=Flatten(name="flatten")(before_model)
before_model=Dense(128,activation="relu")(before_model)
before_model=Dropout(0.5)(before_model)
before_model=Dense(2,activation="softmax")(before_model)

#%%

# We created the model we will use. Now our model will be as follows.

model=Model(inputs=mobile_net_v2.input,outputs=before_model)
model.compile(loss="binary_crossentropy", optimizer="Adam",metrics=["accuracy"])

#%%

# I have to train the model I made. After the training, the model will be ready.

model.fit(image_data_generator.flow(X_train,Y_train),steps_per_epoch=len(X_train)//32,
                      validation_data=(X_test,Y_test),validation_steps=len(X_test)//32,epochs=20)

#%%

# I saved the model in order not to rerun it and to test it on the image we got.

model.save(path+"mask_detection_model.model",save_format="h5")






























