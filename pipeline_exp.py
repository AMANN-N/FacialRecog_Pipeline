from skimage.metrics import structural_similarity as compare_ssim
import os
import cv2
import numpy as np
import warnings
import time
import math
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import torch
from deepface import DeepFace
from facenet_pytorch import MTCNN
import pandas as pd
from src.utility import parse_model_name
warnings.filterwarnings('ignore')
model_dir = "./resources/anti_spoof_models"

device = torch.device("cpu")
class FrameDif:
    def __init__(self,threshold):
        self.threshold = threshold
    def ssim(self,curr_frame,prev_frame):
        current_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        (score,frame_diff) = compare_ssim(previous_frame_gray,current_frame_gray,full = True)
        if (score<self.threshold):
            cv2.putText(curr_frame, 'Change: '+str(score), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #print('Score less than threshold, frame changed '+ str(score))
        else:
            cv2.putText(curr_frame, 'No Change: '+str(score), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return(curr_frame,score)
       
class Antispoof:     
    def predict_spoof(self,parent_frame):
        model_test = AntiSpoofPredict(0)
        image_cropper = CropImage()
        prediction = np.zeros((1, 3))
        image_bbox = model_test.get_bbox(frame)
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": parent_frame,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            label = np.argmax(prediction)
            value = prediction[0][label]
            if label == 1:
                #print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
                result_text = "RealFace Score: {:.2f}".format(value)
                color = (255, 0, 0)
            else:
                #print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
                result_text = "FakeFace Score: {:.2f}".format(value)
                color = (0, 0, 255)
            cv2.putText(parent_frame,result_text,(0,95),cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
         


class Detection:
    def __init__(self,confidence_threshold):
        self.confidence_threshold = confidence_threshold
    def detect(self,frame):
       
        boxes, conf = mtcnn.detect(frame)
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

        # if multiple_faces == True:
        #     print(
        #         str(n_faces)
        #         + " faces detected! "
        #         + "for frame: "
        #         #+ str(frame_number)
        #         + " whose timestamp is: ",
        #         str(cap.get(cv2.CAP_PROP_POS_MSEC)),
        #     )

        # if no_face_detected == True:
        #     print("No face detected!"+ "for frame: "+ '''str(frame_number)'''+ " whose timestamp is: ",str(cap.get(cv2.CAP_PROP_POS_MSEC)),)
        n_faces_str = "No. of faces = " + str(n_faces)
        frame_number_str = "Frame No. = " + 'Test'
        time_stamp_str = "Timestamp = " + str(cap.get(cv2.CAP_PROP_POS_MSEC))
        if conf[0] != None:
            for (x, y, w, h) in boxes:
                text = f"{conf[0]*100:.2f}%"
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.putText(frame,text,(x, y - 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),1)
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 1)

        cv2.putText(frame,frame_number_str,(0, 15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),1,)

        cv2.putText(frame,time_stamp_str,(0, 30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),1,)

        cv2.putText(frame,n_faces_str,(0, 45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),1,)
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
frame_skip = 1
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
antispoof = Antispoof()
recogniser.verify(cv2.resize(curr_frame,(300,300)),curr_frame)
count = 0
count_frame = 0
output_list = []
start_time = time.time()



while True:
    try: 
        frame_input_time = time.time()
        frame_diff_called = 1
        face_detect_called = 0
        face_recog_called = 0
        frame,score =framedif.ssim(curr_frame,prev_frame)

        #detection
        if score<framedif.threshold:
            boxes,nfaces = detector.detect(frame)
            face_detect_called = 1
            print(nfaces)
            if(nfaces==1):
                (x,y,w,h) = boxes[0]
                x, y, w, h = int(x), int(y), int(w), int(h)
                face_frame = frame[y-50:h+50,x-50:w+50]
                #Frame Recognition
                face_recognised = recogniser.verify(cv2.resize(face_frame,(300,300)),frame)
                face_recog_called = 1
                # if(face_recognised):
                #     antispoof.predict_spoof(face_frame,frame)
            if(nfaces>1):
                cv2.putText(frame, "Multiple Faces in Frame", (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                print('ERROR: MULTIPLE FACES IN FRAME')

        cv2.imshow('frame',frame)
        prev_frame = curr_frame.copy() 
        processing_time = time.time()
        time_diff = processing_time - frame_input_time
        time_total = processing_time -start_time
        print("time_diff: "+ str(time_diff) + "Models Called "+ str(frame_diff_called) + " "  +str(face_detect_called)+ " "+ str(face_recog_called))
        output_list.append([time_total,time_diff,frame_diff_called,face_detect_called,face_recog_called])
        #Read next frame
        ret, curr_frame = cap.read()

        count_frame+=1
        print(count_frame)
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
df = pd.DataFrame(data=output_list,columns=[["Timestamp","Processing Time","Frame Difference Called","Face Detection Called","Face Recognition Called"]])
print(df)
df.to_csv("pipeline_output_4.csv")
print('I am out of loop ')