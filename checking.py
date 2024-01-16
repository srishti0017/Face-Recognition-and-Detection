import cv2
import numpy as np

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
        
        id, pred = clf.predict(gray_img[y:y+h,x:x+w])
        confidence = int(100*(1-pred/300))
        
        if confidence>80:
            if id==1:
                cv2.putText(img, "Person 1", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif id==2:
                cv2.putText(img, "Person 2", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            
        else:
            cv2.putText(img, "UNKNOWN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    
    return img



# loading classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# initialize video capture with default camera (index 0)
video_capture = cv2.VideoCapture(0)

while True:
    # read frame from video capture
    ret, img = video_capture.read()
    
    # check if frame was read successfully
    if not ret:
        print("Error reading video capture.")
        break
    
    # skip empty frames
    if img is None:
        continue
    
    # perform face detection
    img = draw_boundary(img, faceCascade, 1.3, 6, (255,255,255), "Face", clf)
    
    # show frame in window
    cv2.imshow("face Detection", img)
    
    # exit on "Enter" key
    if cv2.waitKey(1)==13:
        break

# release video capture and destroy window
video_capture.release()
cv2.destroyAllWindows()
