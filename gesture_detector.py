import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
import threading

MODEL_PATH = os.path.join(
    "prepoznava_geste", "modeli", "gesture_model.pkl"
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# naloži model in odpri kamero ob zagonu
print("Nalagam gesture model...")
model = joblib.load(MODEL_PATH)

print("Odpiranje kamere...")
_cap = cv2.VideoCapture(0)
# preberi par frameov da se kamera inicializira
for _ in range(5):
    _cap.read()
print("Kamera pripravljena.")


def normalize_landmarks(landmarks):
    landmarks = landmarks - landmarks[0]
    scale = np.max(np.linalg.norm(landmarks, axis=1))
    return landmarks, scale


def detect_gesture(timeout=30):

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    start = time.time()
    best_label = None
    best_conf = 0

    while time.time() - start < timeout:

        success, frame = _cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label_text = "Iskanje roke..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                landmarks = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in hand_landmarks.landmark
                ])

                landmarks, scale = normalize_landmarks(landmarks)

                if scale < 0.25:
                    continue

                landmarks /= scale
                features = landmarks.flatten()

                proba = model.predict_proba(features.reshape(1, -1))[0]
                idx = np.argmax(proba)
                confidence = proba[idx]
                current_label = model.classes_[idx]

                label_text = f"{current_label} ({confidence*100:.0f}%)"

                if confidence > best_conf:
                    best_conf = confidence
                    best_label = current_label

        preostalo = int(timeout - (time.time() - start))
        cv2.putText(frame, label_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Cas: {preostalo}s", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if best_label:
            cv2.putText(frame, f"Najboljse: {best_label} ({best_conf*100:.0f}%)",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    if best_conf > 0.6:
        return best_label

    return None