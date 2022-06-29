print("Flag=1 means not found in database");
print("Flag =0 means found");

#VGG FACE

from deepface import DeepFace
import cv2
import numpy as np

vidcap = cv2.VideoCapture('new.mp4')
vidcap.set(cv2.CAP_PROP_POS_MSEC,6000)      # just cue to 5 sec. position
success,image = vidcap.read()
if success:
    cv2.imwrite("framesec.jpg", image)     # save frame as JPEG file

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace",  "Dlib"]
df = DeepFace.find(img_path = "framesec.jpg", db_path = "C:/Users/AMAN RAJ SINGH/Downloads/Facedetection/newframes", model_name = models[0] , enforce_detection= False)

d = 0;
flag = 0;
if df.empty:
    flag = 1 ;
else:

    len = df.shape[0]
    for i in range(len):

        df1 =  np.array(df)
        k  = df1[i][1]

        if (k >= 0.15) :
            d=d+1;

    if(d>1):
        flag=1;
    else:
        flag=0;




print(flag);



#FACE NET
from deepface import DeepFace
import cv2

vidcap = cv2.VideoCapture('new.mp4')
vidcap.set(cv2.CAP_PROP_POS_MSEC,6000)      # just cue to 5 sec. position
success,image = vidcap.read()
if success:
    cv2.imwrite("framesec.jpg", image)     # save frame as JPEG file

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace",  "Dlib"]
df = DeepFace.find(img_path = "framesec.jpg", db_path = "C:/Users/AMAN RAJ SINGH/Downloads/Facedetection/newframes", model_name = models[1] , enforce_detection= False)

d = 0;
flag = 0;
if df.empty:
    flag = 1 ;
else:

    len = df.shape[0]
    for i in range(len):

        df1 =  np.array(df)
        k  = df1[i][1]

        if (k >= 0.15) :
            d=d+1;

    if(d>1):
        flag=1;
    else:
        flag=0;


print(flag);




#OPEN FACE

from deepface import DeepFace
import cv2

vidcap = cv2.VideoCapture('new.mp4')
vidcap.set(cv2.CAP_PROP_POS_MSEC,6000)      # just cue to 5 sec. position
success,image = vidcap.read()
if success:
    cv2.imwrite("framesec.jpg", image)     # save frame as JPEG file

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace",  "Dlib"]
df = DeepFace.find(img_path = "framesec.jpg", db_path = "C:/Users/AMAN RAJ SINGH/Downloads/Facedetection/newframes", model_name = models[2] , enforce_detection= False)

d = 0;
flag = 0;
if df.empty:
    flag = 1 ;
else:

    len = df.shape[0]
    for i in range(len):

        df1 =  np.array(df)
        k  = df1[i][1]

        if (k >= 0.15) :
            d=d+1;

    if(d>1):
        flag=1;
    else:
        flag=0;


print(flag);



#DEEP FACE

from deepface import DeepFace
import cv2

vidcap = cv2.VideoCapture('new.mp4')
vidcap.set(cv2.CAP_PROP_POS_MSEC,6000)      # just cue to 5 sec. position
success,image = vidcap.read()
if success:
    cv2.imwrite("framesec.jpg", image)     # save frame as JPEG file

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace",  "Dlib"]
df = DeepFace.find(img_path = "framesec.jpg", db_path = "C:/Users/AMAN RAJ SINGH/Downloads/Facedetection/newframes", model_name = models[3] , enforce_detection= False)

d = 0;
flag = 0;
if df.empty:
    flag = 1 ;
else:

    len = df.shape[0]
    for i in range(len):

        df1 =  np.array(df)
        k  = df1[i][1]

        if (k >= 0.15) :
            d=d+1;

    if(d>1):
        flag=1;
    else:
        flag=0;


print(flag);



#DLIB
from deepface import DeepFace
import cv2

vidcap = cv2.VideoCapture('new.mp4')
vidcap.set(cv2.CAP_PROP_POS_MSEC,6000)      # just cue to 5 sec. position
success,image = vidcap.read()
if success:
    cv2.imwrite("framesec.jpg", image)     # save frame as JPEG file

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace",  "Dlib"]
df = DeepFace.find(img_path = "framesec.jpg", db_path = "C:/Users/AMAN RAJ SINGH/Downloads/Facedetection/newframes", model_name = models[4] , enforce_detection= False)

d = 0;
flag = 0;
if df.empty:
    flag = 1 ;
else:

    len = df.shape[0]
    for i in range(len):

        df1 =  np.array(df)
        k  = df1[i][1]

        if (k >= 0.15) :
            d=d+1;

    if(d>1):
        flag=1;
    else:
        flag=0;


print(flag);
