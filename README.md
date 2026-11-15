# Drowsiness Detector

Application de détection de somnolence en temps réel par vision par ordinateur, conçue pour analyser l'attention du conducteur via un flux webcam.

## 🚀 Fonctionnalités
- **Détection en temps réel** : Suivi du visage et des yeux image par image via OpenCV.
- **Interface Web interactive** : Pilotage de l'application et visualisation du flux vidéo grâce à **Streamlit**.
- **Gestion anti-faux positifs** : Système de comptage de frames consécutives pour tolérer les clignements naturels.
- **Alertes multimodales** : Affichage d'un message d'avertissement visuel sur le flux et déclenchement d'un signal sonore natif (`winsound`).

---

## 🛠️ Stack Technique
- **Langage** : Python
- **Interface graphique** : Streamlit
- **Traitement d'images** : OpenCV (`cv2`)
- **Calculs numériques** : NumPy

---

## 📦 Installation et Lancement

**Cloner le dépôt :**
   ```bash
   git clone [https://github.com/oyeolouwaa/Drowsiness_Detector.git](https://github.com/oyeolouwaa/Drowsiness_Detector.git)
   cd Drowsiness_Detector

---

## Structure

Drowsiness_Detector/
├── models/
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
├── src/
│   ├── app.py
│   └── detector.py
├── requirements.txt
└── README.md
