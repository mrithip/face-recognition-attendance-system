import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import pickle

# Initialize MTCNN (for face detection) and Inception Resnet (for face embedding)
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Dataset path and embeddings path
datasets = "dataset"
embeddings_path = 'student_embeddings.pkl'

# Dictionary to store student embeddings
student_embeddings = {}

# Function to generate embeddings for each student
def generate_embeddings():
    for subdir in os.listdir(datasets):
        student_name = subdir
        student_folder = os.path.join(datasets, subdir)

        student_embeddings[student_name] = []

        for filename in os.listdir(student_folder):
            img_path = os.path.join(student_folder, filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces
            boxes, probs = mtcnn.detect(img_rgb)

            if boxes is not None:
                for box in boxes:
                    # Extract the face using MTCNN
                    faces = mtcnn(img_rgb)
                    for face in faces:
                        # Get the embedding for the face
                        embedding = model(face.unsqueeze(0)).detach().cpu().numpy()
                        student_embeddings[student_name].append(embedding)

    # Save embeddings to a file for future use
    with open(embeddings_path, 'wb') as f:
        pickle.dump(student_embeddings, f)

# Generate embeddings for all students
generate_embeddings()
print("Embeddings generated and saved!")
