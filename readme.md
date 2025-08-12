# Face Recognition Attendance System

A Python-based attendance system that uses facial recognition to automatically mark attendance of students.

## Features

- Face detection and recognition using MTCNN and FaceNet
- Real-time attendance marking
- Dataset collection for new students
- Automatic CSV generation for attendance records
- Support for both CPU and GPU processing

## Requirements

```python
opencv-python
numpy
pandas
pytorch
facenet-pytorch
retinaface
```

## Project Structure

- `collecting_dataset.py` - Captures and saves face images for new students
- `mtcnn_training.py` - Generates facial embeddings from collected images
- `mark_attendance.py` - Real-time attendance marking system
- `attendance.csv` - Stores attendance records

## Setup and Usage

1. Install dependencies:
```bash
pip install opencv-python numpy pandas torch facenet-pytorch retinaface
```

2. Add new students:
   - Run `collecting_dataset.py`
   - Enter student name when prompted
   - The script will capture 100 face images automatically
   - Press 'q' to quit early

3. Generate embeddings:
   - Run `mtcnn_training.py`
   - This will create `student_embeddings.pkl` with facial features

4. Mark attendance:
   - Run `mark_attendance.py`
   - System will recognize faces and mark attendance
   - Attendance is saved in `attendance.csv`
   - Press 'q' to exit

## Technical Details

- Face detection: MTCNN (Multi-task Cascaded Convolutional Networks)
- Face recognition: InceptionResnetV1 (pretrained on VGGFace2)
- Image processing: OpenCV
- Data storage: Pickle for embeddings, CSV for attendance

## File Descriptions

- `collecting_dataset.py`: Uses RetinaFace for face detection and captures 100 images per student
- `mtcnn_training.py`: Processes collected images to generate facial embeddings
- `mark_attendance.py`: Real-time face recognition and attendance marking system

## Output

The system generates `attendance.csv` with the following columns:
- Name
- Time (YYYY-MM-DD HH:MM:SS)

## Notes

- The system uses webcam index 1 by default (can be changed in `mark_attendance.py`)
- Face recognition threshold is set to 0.9 (can be adjusted for stricter/looser matching)
- Images are stored in the `dataset` folder, organized by student name