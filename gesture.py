
import cv2
import mediapipe as mp


# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, hand_label):
    """
    Count extended fingers using hand landmarks.
    The webcam frame is mirrored before detection.
    """

    lm = hand_landmarks.landmark
    fingers = []

    # Thumb: check horizontal position of tip and joint.
    # Hand labels are supplied by MediaPipe.
    if hand_label == "Right":
        thumb_up = lm[4].x < lm[3].x
    else:
        thumb_up = lm[4].x > lm[3].x

    fingers.append(1 if thumb_up else 0)

    # Index, middle, ring and little fingers.
    # A raised fingertip is above its PIP joint.
    finger_pairs = [
        (8, 6),    # Index
        (12, 10),  # Middle
        (16, 14),  # Ring
        (20, 18)   # Little
    ]

    for tip_id, pip_id in finger_pairs:
        if lm[tip_id].y < lm[pip_id].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers), fingers


def get_gesture(count):
    gestures = {
        0: "FIST - 0",
        1: "ONE FINGER - 1",
        2: "TWO FINGERS - 2",
        3: "THREE FINGERS - 3",
        4: "FOUR FINGERS - 4",
        5: "OPEN PALM - 5"
    }

    return gestures.get(count, "UNKNOWN")


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        # Try the default backend if DirectShow is unavailable.
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam could not be opened.")
        print("Check Windows camera permissions.")
        return

    # Optional: request a reasonable webcam resolution.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Hand Gesture Recognition started!")
    print("Show your palm clearly to the webcam.")
    print("Keep your fingers separated.")
    print("Press ESC to exit.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        ) as hands:

            while True:
                success, frame = cap.read()

                if not success or frame is None:
                    print("Could not read webcam frame.")
                    break

                # Mirror the image for a selfie-style view.
                frame = cv2.flip(frame, 1)

                # MediaPipe requires RGB input.
                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                gesture_text = "Show your hand"
                finger_count = 0

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    ):
                        # Draw the 21 landmarks and connections.
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                        hand_label = (
                            handedness.classification[0].label
                        )

                        finger_count, finger_states = count_fingers(
                            hand_landmarks,
                            hand_label
                        )

                        gesture_text = get_gesture(finger_count)

                        # Display hand information.
                        cv2.putText(
                            frame,
                            f"Hand: {hand_label}",
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Fingers: {finger_count}",
                            (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2
                        )

                        break

                # Display the recognized gesture.
                cv2.rectangle(
                    frame, (10, 10), (470, 65),
                    (35, 35, 35), -1
                )

                cv2.putText(
                    frame,
                    gesture_text,
                    (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0) if results.multi_hand_landmarks
                    else (0, 165, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(
                    "MediaPipe Hand Gesture Recognition",
                    frame
                )

                # ESC key closes the application.
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Hand Gesture Recognition stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    except ImportError as error:
        print("Missing library:", error)
        print("Install with: python -m pip install mediapipe opencv-python")
    except Exception as error:
        print(f"Unexpected error: {error}")
    finally:
        cv2.destroyAllWindows()
