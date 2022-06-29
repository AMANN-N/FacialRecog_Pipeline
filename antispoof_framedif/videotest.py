import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')
cascPath = '/home/sidk_1023/spoofing_detection/python_scripts/haarcascade_frontalface_default.xml'
# def check_image(image):
#     height, width, channel = image.shape
#     if width/height != 3/4:
#         print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
#         return False
#     else:
#         return True
print('hi')
faceCascade = cv2.CascadeClassifier(cascPath)
cap = cv2.VideoCapture("/home/sidk_1023/Videos/Webcam/WhatsApp Video 2022-02-20 at 12.46.21.mp4")

model_dir = "./resources/anti_spoof_models"

    

while(True):
    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    #frame = cv2.resize(frame, (600, 400))
    # center = [frame.shape[0]/2,frame.shape[1]/2]
    # x = center[1] - 300/2
    # y = center[0] - 400/2
    # frame = frame[y:y+400, x:x+300]
    # result = check_image(frame)
    # if result is False:
    #     break
    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
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
    #print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        frame,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        frame,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*frame.shape[0]/1024, color)

    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()