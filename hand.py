
import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingers = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm = hand_landmarks.landmark

            # Thumb detection
            if lm[4].x < lm[3].x:
                fingers += 1

            # Other four fingers
            tips = [8, 12, 16, 20]
            joints = [6, 10, 14, 18]

            for tip, joint in zip(tips, joints):
                if lm[tip].y < lm[joint].y:
                    fingers += 1

        # Display finger count
        cv2.putText(
            frame,
            "Fingers: " + str(fingers),
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    else:
        cv2.putText(
            frame,
            "Show your hand",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Hand Gesture Recognition", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
