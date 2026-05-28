import cv2
import mediapipe as mp
import csv
import numpy as np


# MediaPipe inicializacija

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# CSV dataset file

file = open("dataset/surfer_dataset.csv", mode="a", newline="")
writer = csv.writer(file)

# Trenutna labela
current_label = None

# Counters
frame_count = 0
sample_count = 0

# Kamera

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def normalize_landmarks(landmarks):
    landmarks = landmarks - landmarks[0]
    scale = np.max(np.linalg.norm(landmarks, axis=1))
    return landmarks, scale

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Performance optimization
    rgb_frame.flags.writeable = False

    # Hand detection
    results = hands.process(rgb_frame)

    rgb_frame.flags.writeable = True

    # Landmark processing

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            #landmarks -> numpy
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            #normalizacija
            landmarks, scale = normalize_landmarks(landmarks)

            #ignoriranje oddaljene roke
            if scale < 0.25:
                cv2.putText(
                    frame,
                    "HAND TOO FAR",
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                continue

            # scale normalizacija
            landmarks = landmarks / scale

            # Feature extraction
            features = landmarks.flatten().tolist()

            frame_count += 1

            #shrani vsaki 5. frame
            if current_label is not None and frame_count % 5 == 0:
                row = features + [current_label]
                writer.writerow(row)
                sample_count += 1
                print(f"Saved samples: {sample_count}")

    # UI text
    cv2.putText(
        frame,
        f"Recording: {current_label}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "1:OK  2:PEACE  3:SURFER  4:ALIEN  5:HEART  0:STOP",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show frame

    cv2.imshow("Gesture Dataset Recorder", frame)

    # =========================
    # Keyboard controls
    # =========================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        current_label = "ok"
        print("Recording OK")

    elif key == ord('2'):
        current_label = "peace"
        print("Recording PEACE")

    elif key == ord('3'):
        current_label = "surfer"
        print("Recording SURFER")

    elif key == ord('4'):
        current_label = "alien"
        print("Recording ALIEN")

    elif key == ord('5'):
        current_label = "korean_heart"
        print("Recording KOREAN HEART")

    elif key == ord('0'):
        current_label = None
        print("Recording stopped")

    elif key == 27:  # ESC
        break

# Cleanup

file.close()

cap.release()
cv2.destroyAllWindows()