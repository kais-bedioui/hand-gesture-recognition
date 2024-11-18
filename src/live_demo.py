import numpy as np
import cv2
import time
import math

from hand_sign_recognition.hand_detector import HandDetector
from hand_sign_recognition.hand_signs_recogniser import HandSignsRecogniser
from utils.landmarks_visualiser import draw_landmarks_on_image

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
HANDS = ["Right", "Left"]

# One-hot-encoding of target hand signs
# The format of the data is as follows:
# {"hand_sign": [Thumb, Index, Middle, Ring, Pinky]}
# True: Finger is extended, False: Finger is folded (Not Extended) 
HAND_SIGNS = {
    "open hand": [True, True, True, True, True],
    "peace sign": [False, True, True, False, False],
    "surf sign": [True, False, False, False, True],
    "ok sign": [False, False, True, True, True]
}

def main(model_asset_path="hand_landmarker.task", num_hands=2, stream=True):
    hand_detector = HandDetector(model_asset_path=model_asset_path, num_hands=num_hands, stream=stream)
    hand_sign_recogniser = HandSignsRecogniser()
    cap = cv2.VideoCapture(0)
    # Main Loop
    while True:
        # capture image
        ret, frame = cap.read()

        if ret:
            hand_landmarker_results = hand_detector.get_default_detections(frame)
            hand_signs = hand_sign_recogniser.hand_gestures_recognition(hand_landmarker_results)
            annotated_image = draw_landmarks_on_image(frame, hand_landmarker_results, hand_signs=hand_signs)

            cv2.imshow('Live Demo', annotated_image)
        else:
            print("! No frame")

        time.sleep(0.05)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()

    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(
        model_asset_path="../data/models/hand_landmarker.task",
        num_hands=2,
        stream=True
    )
