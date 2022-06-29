print("Flag = 1 means not found in database");
print("Flag = 0 means found");

#VGG FACE

from deepface import DeepFace
import cv2
import numpy as np


time = 12000;
vidcap = cv2.VideoCapture('vid.mp4')
vidcap.set(cv2.CAP_PROP_POS_MSEC,time)      # just cue to position
success,image = vidcap.read()
if success:
    cv2.imwrite("framesec.jpg", image)     # save frame as JPEG file

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace",  "Dlib"]
df1 = DeepFace.verify(img1_path = "framesec.jpg", img2_path = "database/4.jpg", model_name = models[0])
first_value = list(df1.values())[0]

flag = 0;
if(first_value == True):
    flag = 0;
else:
    flag = 1;
#df = DeepFace.find(img_path = "framesec.jpg", db_path = "C:/Users/AMAN RAJ SINGH/Downloads/Facedetection/database", model_name = models[0] , enforce_detection= False)

#d = 0;
#if df.empty:
#    flag = 1;
#else:

#    len = df.shape[0]
#    for i in range(len):
#
#        df1 =  np.array(df)
#        k  = df1[i][1]
#
#        if (k >= 0.15) :
#            d=d+1;
##    if(d>1):
#        flag=1;
#    else:
#        flag=0;

print(flag)
time_new = time/1000
if flag == 1:
    print("Face not recognized")
else:
    print("Face at time  " , time_new , "  seconds, was recognized")

image = cv2.resize(image, (600, 400))
op= "frame_at_time"+ str(time_new) + "seconds"
cv2.putText(
                image,
                op,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0, 255),
                1,
            )
cv2.imshow("Frame", image)
cv2.waitKey(0)
