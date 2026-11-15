from scipy.spatial import distance as dist
import cv2
import mediapipe as mp
import numpy as np


def eye_aspect_ratio(eye):
  # Calcul des distances euclidiennes entre les deux ensembles de repères verticaux
  A = dist.euclidean(eye[1], eye[5])
  B = dist.euclidean(eye[2], eye[4])

  # Calcul de la distance euclidienne entre les repères horizontaux
  C = dist.euclidean(eye[0], eye[3])

  # Calcul du ratio EAR
  ear = (A + B) / (2.0 * C)
  return ear


# Indices des points clés des yeux dans le maillage MediaPipe Face Mesh
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 144, 153]

# Seuils de détection de la fatigue
EAR_THRESHOLD = 0.22
EAR_CONSEC_FRAMES = 48  # Nombre de frames consécutives sous le seuil pour déclencher l'alerte

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
counter = 0
alarm_on = False

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  h, w, _ = frame.shape
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = face_mesh.process(rgb_frame)

  if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0].landmark

    # Extraction des coordonnées des yeux
    left_eye = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in LEFT_EYE_INDICES]
    )
    right_eye = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in RIGHT_EYE_INDICES]
    )

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    # Moyenne des deux yeux
    ear = (left_ear + right_ear) / 2.0

    # Affichage du EAR sur le flux vidéo
    cv2.putText(
        frame,
        f"EAR: {ear:.2f}",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    if ear < EAR_THRESHOLD:
      counter += 1
      if counter >= EAR_CONSEC_FRAMES:
        cv2.putText(
            frame,
            "ALERTE SOMNOLENCE !",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )
    else:
      counter = 0

  cv2.imshow("Detection de somnolence", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()