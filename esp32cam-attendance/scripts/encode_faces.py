import face_recognition
import cv2
import os
import pickle

path = "../dataset"
encodeList = []
classNames = []

for person in os.listdir(path):
    person_path = os.path.join(path, person)
    if not os.path.isdir(person_path):
        continue
    for file in os.listdir(person_path):
        img = cv2.imread(os.path.join(person_path, file))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)
        if enc:
            encodeList.append(enc[0])
            classNames.append(person)

with open("../encodings.pkl", "wb") as f:
    pickle.dump((encodeList, classNames), f)

print("Encodings saved to encodings.pkl")
