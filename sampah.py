from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('model sampah.h5')

cap=cv2.VideoCapture(0)

labels_dict={0:'Buang',1:'Daur Ulang'}

while(True):

    ret,frame=cap.read()
    frame = cv2.flip(frame,1)
    copy = frame.copy()
    cv2.rectangle(copy, (320,100), (620,400), (255,255,255), 5)
    roi = frame[100:400,320:620]
    roi = cv2.resize(roi,(100,100), interpolation=cv2.INTER_AREA)
    roi = roi/255
    roi = np.reshape(roi,(1,100,100,3))
    result = model.predict(roi) 
    label=np.argmax(result,axis=1)[0]
    cv2.putText(copy, labels_dict[label], (280, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
  
    cv2.imshow('Frame',copy)
    key=cv2.waitKey(1)
    
    if(key==13):
        break
cap.release()        
cv2.destroyAllWindows()