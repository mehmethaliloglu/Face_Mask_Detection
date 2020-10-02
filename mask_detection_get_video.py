# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:56:31 2020

@author: monster
"""

# Libraries

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import time
import cv2

#%%

# I have defined the files that will be used to detect faces in the image.

path="C:/Users/monster/Desktop/mask_detection_project/"
deploy = path+"deploy.prototxt.txt"
caffe_model=path+"res10_300x300_ssd_iter_140000.caffemodel"
face_recognition=cv2.dnn.readNet(deploy,caffe_model)

#%%

# I will use the model we created for mask estimation here.

mask_model=load_model(path+"mask_detection_model.model")

#%%

# We will read the video and estimate the frames taken from there with mask_model.

# If you want to capture images from your own camera, use 'camera = cv2.VideoCapture(0)'.
#camera = cv2.VideoCapture(0)
camera=cv2.VideoCapture("mask_control.mp4")
time.sleep(2.0)
while(camera.isOpened()):
    _ , frame=camera.read()
    frame=imutils.resize(frame,width=800) # I resized the video.
    (height,width)=frame.shape[:2] # I kept the Height and Width of the resized video.
    
    #I did average subtraction, scaling and resizing the image.
    blob_from_image=cv2.dnn.blobFromImage(frame,1.0,(210,210),(104.0,177.0,123.0)) 
    
    #I gave the frame I processed in blob_from_image to the face_recognition model that I created for face recognition.
    face_recognition.setInput(blob_from_image)
    finding=face_recognition.forward() 
    
    # I kept the position and array of each face I found and the predictions I found.
    face_list=[]
    locasion_list=[]
    prediction_list=[]
    
    
    for i in range(0,finding.shape[2]):
        # The minimum probability to filter weak face detections.By default, this value is 30-50%.
        threshold=finding[0,0,i,2] 
        
        if (threshold>0.3):
            
            # Since the first 2 elements of Finding are 0-1 and the third element is threshold,
            # I started dimensioning from the 4th (ie 3rd index) element.
            # I got 4 elements between 3 and 7. I multiplied these by the Width-Height I found.
            box=finding[0,0,i,3:7]*np.array([width,height,width,height])
            
            # I converted the float box values ​​to int. I have assigned the start and end values ​​to the variable.
            start_width,start_height,end_width,end_height=box.astype("int")

            start_width,start_height=max(0,start_width),max(0, start_height)
            end_width,end_height=min(width,end_width),min(height,end_height)
            
            # I cut the face in the video from the start and end values ​​I found.
            face_image=frame[start_height:end_height,start_width:end_width]
            
            # When I was training the model, the size of the images was 210x210. So I resized the image I cut.
            face_image=cv2.resize(face_image,(210,210),interpolation=cv2.INTER_CUBIC)
            
            #I converted it to numpy array.
            face_image=img_to_array(face_image)
            face_image=preprocess_input(face_image)
            
            face_list.append(face_image)
            locasion_list.append((start_width,start_height,end_width,end_height))
            
    if(len(face_list)>0):
        face_list=np.array(face_list,dtype="float32")
        prediction_list=mask_model.predict(face_list,batch_size=32)
        
        # Guessing is over. Now we will add the result to the image.
        
    for box,prediction in zip(locasion_list,prediction_list):
            
        start_width,start_height,end_width,end_height=box
            
        masked,unmasked=prediction
            
        if(masked>unmasked):
            label="MASKED"
            color=(0,255,0)
        else:
            label="UNMASKED"
            color=(0,0,255)
                
        text=label+":"+str(round(max(masked, unmasked) * 100,2))
            
        cv2.putText(frame,text,(start_width,start_height-10),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,thickness=2,color=color)
        cv2.rectangle(frame,(start_width,start_height),(end_width,end_height),color=color,thickness=2)
        
    cv2.imshow("Mask Predict",frame)
        
    key=cv2.waitKey(10) & 0xFF 

    if(key==ord("q")):
        break
camera.release()
cv2.destroyAllWindows()
