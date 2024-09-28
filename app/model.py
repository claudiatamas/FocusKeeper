from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
from joblib import dump
import os
import mediapipe as mp


face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



def extract_eye_regions(image, landmarks, eye_indices):
    eye_features = []
    for index in eye_indices:
        landmark = landmarks.landmark[index]
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        eye_features.extend([x, y])
    return np.array(eye_features)


def load_images_from_folder(folder, label, face_mesh):
    data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_features = extract_eye_regions(img_rgb, face_landmarks, LEFT_EYE_INDICES)
                right_eye_features = extract_eye_regions(img_rgb, face_landmarks, RIGHT_EYE_INDICES)
                eye_features = np.concatenate((left_eye_features, right_eye_features))
                data.append((eye_features, label))
    return data



LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

active_data = load_images_from_folder('./static/faceimages/ActiveSubjects', 0, face_mesh)
fatigue_data = load_images_from_folder('./static/faceimages/FatigueSubjects', 1, face_mesh)

all_data = active_data + fatigue_data
X, y = zip(*all_data)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model accuracy: {accuracy:.2f}')

dump(model, './static/random_forest_model_train.joblib')
