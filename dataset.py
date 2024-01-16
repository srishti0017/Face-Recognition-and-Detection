import cv2
import os

def generate_dataset():
    # face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_classifier = cv2.CascadeClassifier("C:/Users/Lenovo/Desktop/AI Project/2nd Part/haarcascade_frontalface_default.xml")
    if face_classifier.empty():
        raise ValueError("Unable to load face classifier")
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Unable to open video capture device")
    
    id = 1
    img_id = 0
    max_images = 200
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from video capture device")
            continue
        
        cropped_face = face_cropped(frame, face_classifier)
        if cropped_face is None:
            continue
        
        img_id += 1
        face = cv2.resize(cropped_face, (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file_name_path = f"C:/Users/Lenovo/Desktop/AI Project/2nd Part/data/user.{id}.{img_id}.jpg"
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Cropped face", face)
            
        if cv2.waitKey(1) == ord('q') or img_id == max_images:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed....")
    

def face_cropped(img, face_classifier):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    
    (x, y, w, h) = faces[0]
    cropped_face = img[y:y+h, x:x+w]
    return cropped_face
    

if __name__ == "__main__":
    generate_dataset()
