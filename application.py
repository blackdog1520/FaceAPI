from flask import Flask,request

import cv2
import numpy as np
import os
import imutils
import requests
import json
import shutil
import sys
from firebase import Firebase








application = app = Flask(__name__)
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()



def index(name,verify):
    test_imgnew = cv2.imread(name)
    #rotation angle in degree
    
    faces_detected,gray_imag = faceDetection(test_imgnew)
    print("faces ddetected:",faces_detected)


    #faces,faceID = labels_for_training_data('trainingImage')					#trainingImage is the folder where you need to save you trainings images of each person in a file inside it.
    #face_recognizer =train_classifier(faces,faceID)								#train you model with dataset provided by you and save it as trainingData.yml, and the comment these lines again   
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()							#run  this line when you have saved the trainingData.yml file
    #face_recognizer.save('trainingData.yml')
    
    face_recognizer.read('trainingData.yml')
    
    name = {0:<First person name>,1:<Second Person Name>}
    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_imag[y:y+w,x:x+h]

        resized_imgnew = cv2.resize(roi_gray,(600,600))
        label3,confidence3 = face_recognizer.predict(resized_imgnew)
        if(verify == name[label3]):
            return '200'
        else:
            return '404'

@application.route('/loadImg')
def loadImg():
    config = {										#you can find details in firebase project setting
	  "apiKey": <Firebase api key>,
	  "authDomain": "",
	  "databaseURL": "",
	  "storageBucket": ""
	}
    firebase = Firebase(config)
    storage = firebase.storage()
    
    second = request.args.get('paramter_of_url')				#this paramter_of_url will be passed with the url with the name of file which is to verified
    fullname = second+'.jpg'
    storage.child(<file path>).download(fullname)
    nameOfperson = index(fullname,second)
    return json.dumps({"status":nameOfperson,"id":second})	
	
	
def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('HaarCasacades/haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img,scaleFactor = 1.32, minNeighbors = 7)
    return faces,gray_img
def labels_for_training_data(directory):
    faces= []
    faceID =[]
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping files")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path,filename)
            print("img_path",img_path)
            print("id:",id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded")
                continue
            faces_rect,gray_img= faceDetection(test_img)
            if len(faces_rect)!=1:
                continue
            (x,y,w,h)=faces_rect[0]
            roi_gray = gray_img[y:y+w,x:x+h]
            resized_img = cv2.resize(roi_gray,(600,600))
            faces.append(resized_img)
            faceID.append(int(id))
    return faces,faceID

def train_classifier(faces,faceID):
    #face_recognizer1 = cv2.face.EigenFaceRecognizer_create()
    #face_recognizer1.train(faces,np.array(faceID))
    #face_recognizer2 = cv2.face.FisherFaceRecognizer_create()
    #face_recognizer2.train(faces,np.array(faceID))
    face_recognizer3 = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer3.train(faces,np.array(faceID))
    return face_recognizer3




   