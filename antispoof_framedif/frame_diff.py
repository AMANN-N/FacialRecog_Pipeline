import cv2
from skimage import measure

cap= cv2.VideoCapture("/home/sidk_1023/Videos/Webcam/WhatsApp Video 2022-02-20 at 12.46.21.mp4")
ret, current_frame = cap.read()
previous_frame = current_frame
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    try:
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
        (score,frame_diff) = measure.compare_ssim(previous_frame_gray,current_frame_gray,full = True)
        text = str(score)
        if (score<0.75):
            cv2.putText(frame_diff, 'Change: '+text, (50,50), font, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame_diff, 'No Change: '+text, (50,50), font, 1, (0, 255, 0), 2)
        cv2.imshow('frame diff ',frame_diff)  
        previous_frame = current_frame.copy()
        ret, current_frame = cap.read()    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        break    
cap.release()
cv2.destroyAllWindows()

   

