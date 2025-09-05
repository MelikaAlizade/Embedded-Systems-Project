import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import requests
import numpy as np
import time

# Load encodings
with open("../encodings.pkl", "rb") as f:
    encodeListKnown, classNames = pickle.load(f)

df = pd.DataFrame(columns=['Name', 'Time'])
url = "http://192.168.1.20"


while True:
    resp = requests.get(f"{url}/capture")
    img_arr = np.array(bytearray(resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb)
    encodes = face_recognition.face_encodings(rgb, faces)

    for enc, loc in zip(encodes, faces):
        matches = face_recognition.compare_faces(encodeListKnown, enc)
        faceDis = face_recognition.face_distance(encodeListKnown, enc)
        best = faceDis.argmin()

        if matches[best]:

            name = classNames[best]
            if name not in df['Name'].values:
                df = pd.concat([df, pd.DataFrame(
                    {'Name': [name], 'Time': [datetime.now().strftime('%H:%M:%S')]})])
            y1, x2, y2, x1 = loc
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

df.to_csv("../attendance.csv", index=False)
cv2.destroyAllWindows()
