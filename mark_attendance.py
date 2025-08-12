import cv2
import numpy as np
import pandas as pd
import datetime
import pickle
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Load pre-trained models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load student embeddings
with open('student_embeddings.pkl', 'rb') as f:
    student_embeddings = pickle.load(f)

# Webcam
webcam = cv2.VideoCapture(1)

# Attendance file path
attendance_file = 'attendance.csv'

# Attendance marking function
def mark_attendance(name):
    df = pd.read_csv(attendance_file) if os.path.exists(attendance_file) else pd.DataFrame(columns=['Name', 'Time'])
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if name not in df['Name'].values:
        df.loc[len(df)] = [name, time_stamp]
        df.to_csv(attendance_file, index=False)
        print(f"Attendance marked for {name}")

# Compare embeddings
def compare_embeddings(face_embedding, stored_embeddings):
    best_match = None
    min_distance = float('inf')
    for name, embeddings in stored_embeddings.items():
        for emb in embeddings:
            distance = np.linalg.norm(face_embedding - emb)
            if distance < min_distance:
                min_distance = distance
                best_match = name
    return best_match, min_distance

# Main loop
while True:
    ret, frame = webcam.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get face boxes and cropped face tensors
    boxes, probs = mtcnn.detect(img_rgb)
    faces = mtcnn.extract(img_rgb, boxes, save_path=None) if boxes is not None else []

    if boxes is not None and faces is not None:
        for box, face in zip(boxes, faces):
            # Get embedding
            face_embedding = model(face.unsqueeze(0).to(device)).detach().cpu().numpy()

            # Compare with database
            name, distance = compare_embeddings(face_embedding, student_embeddings)

            if distance < 0.9:
                mark_attendance(name)
                label = name
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            # Draw box and name
            box = [int(b) for b in box]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
