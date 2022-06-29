from skimage.metrics import structural_similarity as compare_ssim
import os
import cv2
import numpy as np
import warnings
import time
import math
import torch
from deepface import DeepFace
from facenet_pytorch import MTCNN
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

device = torch.device("cpu")
class FrameDif:
    def __init__(self,threshold):
        self.threshold = threshold
    def ssim(self,curr_frame,prev_frame):

        #framediff_pass = True
        #cap.set(cv2.CAP_PROP_POS_MSEC,curr_time) 
        #ret,curr_frame = cap.read()
        #cv2.imshow('curr_frame',curr_frame)
        #curr_frame = cv2.resize(curr_frame, (600, 400))
        #cap.set(cv2.CAP_PROP_POS_MSEC,prev_time) 
        #ret,prev_frame = cap.read()
        #prev_frame = cv2.resize(prev_frame, (600, 400))
        current_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame',curr_frame)
        (score,frame_diff) = compare_ssim(previous_frame_gray,current_frame_gray,full = True)
        print('Score is:' + str(score))

        if (score<self.threshold):
            cv2.putText(curr_frame, 'Change: '+str(score), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            print('Score less than threshold, frame changed '+ str(score))
        else:
            cv2.putText(curr_frame, 'No Change: '+str(score), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return(curr_frame,score)
       

class Detection:
    def __init__(self,confidence_threshold):
        self.confidence_threshold = confidence_threshold
    def detect(self,frame):
        # Here we are going to use the facenet detector
        boxes, conf = mtcnn.detect(frame)
        # conf is a list of the confidence values of each detection
        # box is a list of four tuples where each of the tuples
        # contain the x,y,width,height of a box that contains the detected face

        #    print('\n boxes: ', boxes)
        #    print('\n conf: ', conf)
        # Define a confidence threshold:

        multiple_faces_detected = False
        no_face_detected = False
        n_faces = 0

        for i in conf:

            if i == None:
                i = 0
            if i > self.confidence_threshold:
                n_faces += 1

        if n_faces > 1:
            multiple_faces = True
        else:
            multiple_faces = False

        if n_faces == 0:
            no_face_detected = True

        if multiple_faces == True:
            print(
                str(n_faces)
                + " faces detected! "
                + "for frame: "
                #+ str(frame_number)
                + " whose timestamp is: ",
                str(cap.get(cv2.CAP_PROP_POS_MSEC)),
            )

        if no_face_detected == True:
            print(
                "No face detected!"
                + "for frame: "
                #+ str(frame_number)
                + " whose timestamp is: ",
                str(cap.get(cv2.CAP_PROP_POS_MSEC)),
            )

        #frame_number += 1

        n_faces_str = "No. of faces = " + str(n_faces)
        frame_number_str = "Frame No. = " + 'Test'
        time_stamp_str = "Timestamp = " + str(cap.get(cv2.CAP_PROP_POS_MSEC))

        if conf[0] != None:
            for (x, y, w, h) in boxes:
                text = f"{conf[0]*100:.2f}%"
                x, y, w, h = int(x), int(y), int(w), int(h)

                cv2.putText(
                    frame,
                    text,
                    (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 1)

        cv2.putText(
            frame,
            frame_number_str,
            (0, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        cv2.putText(
            frame,
            time_stamp_str,
            (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        cv2.putText(
            frame,
            n_faces_str,
            (0, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        return (boxes,n_faces)

class Recognition:
    def __init__(self,model): #"VGG-Face", "Facenet", "OpenFace", "DeepFace",  "Dlib"
        self.model = model
    def verify(self,frame,parent_frame):
        cv2.imwrite('/home/sidk_1023/Facial_Recognition_Pipeline-main/face_recog_database/framesec.jpg', frame)
        df1 = DeepFace.verify(img1_path = '/home/sidk_1023/Facial_Recognition_Pipeline-main/face_recog_database/framesec.jpg', img2_path = "/home/sidk_1023/Facial_Recognition_Pipeline-main/face_recog_database/first_guy.jpg", model_name = "Facenet",enforce_detection=False)
        score_value = list(df1.values())[1]
        face_recognised= 0
        if(score_value <= 0.025):
            face_recognised = 1
            print("Face was recognized "+ str(score_value))
            cv2.putText(parent_frame, 'Face Recognised '+ str(score_value), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            face_recognised = 0
            
            
            
            print("Face not recognized "+ str(score_value))
            cv2.putText(parent_frame, 'Face Not Recognised '+str(score_value), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return face_recognised

# Create the model
frame_skip = 2
mtcnn = MTCNN(keep_all=True, device=device)
cap= cv2.VideoCapture("/home/sidk_1023/Facial_Recognition_Pipeline-main/face-demographics-walking-and-pause.mp4")
start_time = 0
count = 0
ret, curr_frame = cap.read()
curr_frame = cv2.resize(curr_frame, (600, 400))
print('first frame read')
prev_frame = curr_frame
framedif = FrameDif(0.97)
detector = Detection(0.95)
recogniser = Recognition('VGG-Face')

#framediff_flag = True

count = 0
while True:
    try:
        start_time = time.time()
        frame,score =framedif.ssim(curr_frame,prev_frame)
        #detection
        if score<framedif.threshold:
            boxes,nfaces = detector.detect(frame)
            print(nfaces)
                        #mg[Y:Y+H, X:X+W]
            if(nfaces==1):
                (x,y,w,h) = boxes[0]
                x, y, w, h = int(x), int(y), int(w), int(h)
                face_frame = frame[y-50:h+50,x-50:w+50]
                face_recognised = recogniser.verify(cv2.resize(face_frame,(300,300)),frame)
            if(nfaces>1):
                print('ERROR: MULTIPLE FACES IN FRAME')

        cv2.imshow('frame',frame)
        prev_frame = curr_frame.copy() 
        ret, curr_frame = cap.read()
        count+=frame_skip
        cap.set(cv2.CAP_PROP_POS_FRAMES,count)

        curr_frame = cv2.resize(curr_frame, (600, 400)) 
    except cv2.error:
        break
    #print('frame read from inside the loop')
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
    #prev_frame = curr_frame 
cap.release()
cv2.destroyAllWindows()
  
print('I am out of loop ')