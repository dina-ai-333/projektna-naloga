import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

MODEL_PATH = os.path.join("modeli", "gesture_model.pkl")
model = joblib.load(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_text = "No hand"

def normalize_landmarks(landmarks):
    landmarks = landmarks - landmarks[0]
    scale = np.max(np.linalg.norm(landmarks, axis=1))
    return landmarks, scale

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesture_text = "No hand"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            #normalizacija
            landmarks, scale = normalize_landmarks(landmarks)

            #filter za oddaljeno roko
            if scale < 0.25:
                gesture_text = "Hand too far"
                continue

            #končna normalizacija
            landmarks = landmarks / scale

            features = landmarks.flatten().reshape(1, -1)

            proba = model.predict_proba(features)[0]
            idx = np.argmax(proba)

            confidence = proba[idx]
            predicted_label = model.classes_[idx]

            print("Prediction:", predicted_label)
            print("Confidence:", round(confidence, 3))
            print("Scale:", round(scale, 3))
            print("----------------------------")

            #filter
            if confidence > 0.6:
                gesture_text = predicted_label
            else:
                gesture_text = "unknown"

    cv2.putText(
        frame,
        f"Gesture: {gesture_text}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()