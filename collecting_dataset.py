import cv2
import os
from retinaface import RetinaFace

# Dataset Path
datasets = "dataset"
name = input("Enter student's name: ")
path = os.path.join(datasets, name)

if not os.path.exists(path):
    os.makedirs(path)

# Webcam Initialization
webcam = cv2.VideoCapture(0)
count = 1
(width, height) = (130, 100)

while count <= 100:
    ret, frame = webcam.read()
    if not ret:
        continue

    faces = RetinaFace.detect_faces(frame)

    if faces:
        for key in faces:
            facial_area = faces[key]['facial_area']
            x, y, x_w, y_h = facial_area

            # Draw rectangle & save image
            face = frame[y:y_h, x:x_w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite(f"{path}/{count}.png", face_resize)
            count += 1

    cv2.imshow("Face Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
