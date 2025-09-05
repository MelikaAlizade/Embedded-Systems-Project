import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import requests
import numpy as np
import time
import smtplib
from email.message import EmailMessage
import os

with open("../encodings.pkl", "rb") as f:
    encodeListKnown, classNames = pickle.load(f)

df = pd.DataFrame(columns=['Name', 'Time'])
url = "http://192.168.1.20"

# url = "http://192.168.1.20"
# recognition
# nrgx eixy aicp fjbw
SENDER_EMAIL = "melikaalizadeh0@gmail.com"
APP_PASSWORD = "nrgx eixy aicp fjbw"
RECEIVER_EMAIL = "alizadehmelika369@gmail.com"


def send_email(name, time_str, image_path):
    msg = EmailMessage()
    msg["Subject"] = f"[Attendance] {name} entered at {time_str}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content(f"Person recognized: {name}\nTime: {time_str}")

    # attach image
    with open(image_path, "rb") as f:
        img_data = f.read()
    msg.add_attachment(img_data, maintype="image",
                       subtype="jpeg", filename=os.path.basename(image_path))

    # send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
    print(f"[+] Email sent for {name} at {time_str}")


# Cooldown system
last_sent = {}
cooldown = 30    # seconds between emails per person


while True:
    try:
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
                time_str = datetime.now().strftime('%H:%M:%S')
                now = time.time()

                # check cooldown
                if name not in last_sent or (now - last_sent[name]) > cooldown:
                    last_sent[name] = now

                    # log attendance
                    df = pd.concat(
                        [df, pd.DataFrame({'Name': [name], 'Time': [time_str]})])

                    # save frame
                    os.makedirs("../captures", exist_ok=True)
                    img_filename = f"../captures/{name}_{time_str.replace(':', '-')}.jpg"
                    cv2.imwrite(img_filename, frame)

                    # send email
                    send_email(name, time_str, img_filename)

                # draw bounding box
                y1, x2, y2, x1 = loc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error:", e)
        time.sleep(1)

df.to_csv("../attendance.csv", index=False)
cv2.destroyAllWindows()
