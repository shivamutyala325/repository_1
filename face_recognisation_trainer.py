import cv2
import os
import numpy as np


classifier = cv2.CascadeClassifier(r'C:\Users\shiva\AppData\Roaming\JetBrains\PyCharmCE2023.1\scratches\haar_face.xml')
persons = []
data_dir = r"C:\Users\shiva\Desktop\seperated_faces"


for person in os.listdir(data_dir):
    persons.append(person)

print("Detected Persons:", persons)


features = []
labels = []


def train():
    for person in persons:
        path = os.path.join(data_dir, person)
        label = persons.index(person)

        for face_file in os.listdir(path):
            face_path = os.path.join(path, face_file)

            img = cv2.imread(face_path)
            if img is None:
                print(f"Failed to load image: {face_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face_coords = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in face_coords:
                face_roi = gray[y:y + h, x:x + w]

                features.append(face_roi)
                labels.append(label)


train()

if len(features) == 0 or len(labels) == 0:
    print("No data to train on. Please check the dataset.")
    exit()

print(f"Number of training samples: {len(labels)}")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(features, np.array(labels))

model_path = 'trained_data1.yml'
recognizer.save(model_path)
print(f"Model trained and saved at {model_path}")
