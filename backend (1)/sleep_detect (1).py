### IMPORT LIBRARIES ###
from keras import backend as K
import imutils
from keras.models import load_model
import numpy as np
import keras
import requests
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import dlib
import cv2, os, sys
import collections
import random
import face_recognition
import pickle
import math
from threading import Thread, active_count
import tensorflow as tf
from playsound import playsound
import shutil

import firebase

### SET OUTPUT FILE NAME ###

now = time.asctime( time.localtime(time.time()))
x = str(now).replace(" ", "_") 
x = x.replace(":",".")
f = x+".mp4"

### FACIAL RECOGNITION CLASSES AND FUNCTIONS ###

# Class defines where the eyes are.
class FacialLandMarksPosition:
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Function to see if the eyes are open or not
def predict_eye_state(model, image):
    # Resize images to 20x10
    image = cv2.resize(image, (20, 10))
    image = image.astype(dtype=np.float32)

    # Change images to tensor
    image_batch = np.reshape(image, (1, 10, 20, 1))
    # Use mobilenet to find out if eyes are open or not
    image_batch = keras.applications.mobilenet.preprocess_input(image_batch)

    return np.argmax(model.predict(image_batch)[0])

################ MAIN LOOP ##############################

# Load the dlib model to find all facial landmarks
facial_landmarks_predictor = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(facial_landmarks_predictor)

# Load the model predictors for the eye condition
model = load_model('weights.149-0.01.hdf5')

# Take pictures from the webcam, initialise video variable
cap = cv2.VideoCapture(0)
scale = 0.5
countClose = 0
currState = 0
alarmThreshold = 5
nofacetimeout = 0
isface = 0
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
videoWriter = cv2.VideoWriter(f , fourcc, 30.0, (640,480), True)
buffer = 0
noface = 0

# The main loop

while (True):
    c = time.time()
    # Read the webcam feed and separate them into RGB frames
    ret, frame = cap.read()
    if ret:
        videoWriter.write(frame)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the image to half the original size
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    # Change the color to LAB to get lan brightness
    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    resized_height, resized_width = l.shape[:2]
    height_ratio, width_ratio = original_height / resized_height, original_width / resized_width

    # Find face with HOG
    face_locations = face_recognition.face_locations(l, model='hog')

    # If a face is found then
    if len(face_locations):

        # We get the face locations to create a face square
        top, right, bottom, left = face_locations[0]
        x1, y1, x2, y2 = left, top, right, bottom
        x1 = int(x1 * width_ratio)
        y1 = int(y1 * height_ratio)
        x2 = int(x2 * width_ratio)
        y2 = int(y2 * height_ratio)

        # Get the position of the face

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
        face_landmarks = face_utils.shape_to_np(shape)

        left_eye_indices = face_landmarks[FacialLandMarksPosition.left_eye_start_index:
                                          FacialLandMarksPosition.left_eye_end_index]

        (x, y, w, h) = cv2.boundingRect(np.array([left_eye_indices]))
        left_eye = gray[y:y + h, x:x + w]

        right_eye_indices = face_landmarks[FacialLandMarksPosition.right_eye_start_index:
                                           FacialLandMarksPosition.right_eye_end_index]

        (x, y, w, h) = cv2.boundingRect(np.array([right_eye_indices]))
        right_eye = gray[y:y + h, x:x + w]

        # Use mobilenet to find out if the eye is closed or open 

        left_eye_open = 'yes' if predict_eye_state(model=model, image=left_eye) else 'no'
        right_eye_open = 'yes' if predict_eye_state(model=model, image=right_eye) else 'no'


        # If both eyes are open, create a green square, else a red square
        if left_eye_open == 'yes' and right_eye_open == 'yes':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            currState = 0
            countClose = 0

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            currState = 1
            countClose +=1
        if countClose > alarmThreshold:
            
            noface = 1 # variable for the next part
        else:
            isface = 1 # variable for the next part
    else:
        noface = 1

    frame = cv2.flip(frame, 1)
    
    # Video displaying
        
    cv2.imshow('Sleep Detection', frame)
    
    ### VIDEO CREATION ###
    
    if noface == 1: # the noface variable tracks the amount of frames where the driver's face is not shown or eyes are closed.
        nofacetimeout+=1
    
    if nofacetimeout == 15: # 30 frames of sleep time before we play an airhorn sound and write the file to a local repository
        playsound("veryloudairhorn.mp3", block = False)
        videoWriter.release()
        now = time.asctime( time.localtime(time.time()))
        x = str(now).replace(" ", "_") 
        x = x.replace(":",".")
        f = "./detected/"+"detected_"+x+".mp4"
        f1 = "detected_"+x+".mp4"
        nofacetimeout = 0
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        videoWriter = cv2.VideoWriter(f , cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640,480), True)
        buffer = 0
        #firebase.firebase_upload("detected/"+f1)
        #shutil.copyfile('detected/'+f1, '../frontend/daydianhtaioi/public/videos/'+f1)
#        firebase.firebase_download("../frontend/daydianhtaioi/public/video/"+f1)
       
    # If there is a face, then, reset the other variables to zero. Increment the buffer time, which serves to have time before the video starts. If the buffer exceeds 60 frames, it resets to zero.
    
    if isface == 1:
        nofacetimeout = 0
        isface = 0
        buffer += 1
        if buffer == 60: #buffer time
            fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
            videoWriter = cv2.VideoWriter(f , fourcc, 30.0, (640,480), True);
            buffer = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):  # A force close key
        videoWriter = 0
        break
    
# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()