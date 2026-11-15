import os

import cv2

import numpy as np

import streamlit as st

import winsound



st.title("Détecteur de somnolence en temps réel")

st.write("Analyse de la fatigue par vision par ordinateur.")



run = st.checkbox("Activer la webcam")

FRAME_WINDOW = st.image([])



face_path = os.path.join("models", "haarcascade_frontalface_default.xml")

eye_path = os.path.join("models", "haarcascade_eye.xml")



face_cascade = cv2.CascadeClassifier(face_path)

eye_cascade = cv2.CascadeClassifier(eye_path)



cap = cv2.VideoCapture(0)

counter = 0

# Seuil augmenté à 45 frames (~1,5s) pour éviter les faux positifs sur les clignements normaux

CONSEC_FRAMES_THRESHOLD = 45



while run:

    success, frame = cap.read()

    if not success:

        st.warning("Impossible d'accéder à la webcam.")

        break



    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)



    eyes_detected = False



    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]

        roi_color = frame[y:y+h, x:x+w]



        # On cherche uniquement dans la moitié supérieure du visage pour les yeux

        eye_roi_gray = roi_gray[0:int(h*0.6), :]

        eyes = eye_cascade.detectMultiScale(eye_roi_gray)

       

        if len(eyes) >= 2:

            eyes_detected = True

            for (ex, ey, ew, eh) in eyes:

                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)



    if not eyes_detected and len(faces) > 0:

        counter += 1

        if counter >= CONSEC_FRAMES_THRESHOLD:

            cv2.putText(frame, "ALERTE SOMNOLENCE !", (50, 50),

                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Son d'alerte natif Windows (fréquence 2500 Hz, durée 300 ms)

            try:

                winsound.Beep(2500, 300)

            except RuntimeError:

                pass

    else:

        counter = 0



    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(frame)



cap.release()