######################################
# This script is a simple visualiser of hand landmarks.
# It will use OpenCV to open the Webcam, apply MediaPipe's hand_landmarks model
# and display results
#####################################
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import os
import numpy as np
import cv2
import time
import math

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

def draw_landmarks_on_image(rgb_image, detection_result, hand_signs ={}, margin=10, font_size=1, font_thickness=1, handedness_text_color=(88, 205, 54)):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - margin
        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}-{hand_signs[handedness[0].category_name]}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    font_size, handedness_text_color, font_thickness, cv2.LINE_AA)
    return annotated_image

def extract_finger_landmarks(hand_landmarks=[], finger_index=0, finger_name="Finger"):
    finger_mcp_index = 4*finger_index - 3
    finger_tip_index = 4*finger_index
    finger_landmarks = hand_landmarks[finger_mcp_index:finger_tip_index+1]
    return finger_landmarks

def extract_wrist_landmarks(hand_landmarks=[]):

    if hand_landmarks:
        wrist_landmarks = hand_landmarks[0]
        return wrist_landmarks
    else:
        return []

def finger_is_extended(finger_landmarks, wrist_landmarks, finger="Finger"):
    """
        This function will measure and compare the distances d1=(wrist-finger MCP) & d2=(wrist-finger TIP)
        If d1 > d2: finger folded
        if d2 > d1: finger extended
    """
    if finger_landmarks and wrist_landmarks:
        wrist_x = wrist_landmarks.x
        wrist_y = wrist_landmarks.y

        finger_mcp_x = finger_landmarks[0].x
        finger_mcp_y = finger_landmarks[0].y
        finger_tip_x = finger_landmarks[3].x
        finger_tip_y = finger_landmarks[3].y
        
        if finger.lower() == "thumb":
            dist_wrist_to_mcp = abs(wrist_x - finger_mcp_x)
            dist_wrist_to_tip = abs(wrist_x - finger_tip_x)
        else:
            dist_wrist_to_mcp = math.dist(
                (wrist_x, wrist_y),
                (finger_mcp_x, finger_mcp_y)
            )
            dist_wrist_to_tip = math.dist(
                (wrist_x, wrist_y),
                (finger_tip_x, finger_tip_y)
            )
        
        extended = dist_wrist_to_tip > dist_wrist_to_mcp
        return extended

    else:
        return False

def hand_landmarks_inference(detector, image):
    detection_result = detector.detect(image)
    return detection_result

def hand_gesture_recognition(fingers_extension_vector):

    fingers_extension_vector = np.array(fingers_extension_vector)
    
    for sign, one_hot_vector in HAND_SIGNS.items():
        print(f'{fingers_extension_vector} VS {one_hot_vector}')
        if np.array_equal(fingers_extension_vector, one_hot_vector):
            return sign
    
    print('WARNING - Hand sign unidentified')
    return "Nan"

# Get inference
def get_annotation_from(detector, frame):
    hand_signs = {}
    fingers_extension = []
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    detection_results = hand_landmarks_inference(detector, image)
    handedness = detection_results.handedness ## [[CategoryName], [CategoryName]]
    hand_landmarks = detection_results.hand_landmarks 

    for hand_idx in range(len(hand_landmarks)):
        fingers_extension.append([])
        current_hand_wrist_landmarks = extract_wrist_landmarks(hand_landmarks=hand_landmarks[hand_idx])

        for idx, finger in enumerate(FINGERS):
            
            finger_index = idx + 1
            
            finger_landmarks = extract_finger_landmarks(
                hand_landmarks=hand_landmarks[hand_idx], 
                finger_index=finger_index, 
                finger_name=finger
            )
            
            if finger_landmarks and current_hand_wrist_landmarks:
                is_extended = finger_is_extended(
                    finger_landmarks=finger_landmarks, 
                    wrist_landmarks=current_hand_wrist_landmarks,
                    finger=finger
                )
            
            fingers_extension[hand_idx].append(is_extended)
        
        print(f'{HANDS[hand_idx]}: {fingers_extension[hand_idx]}') 
        hand_signs[handedness[hand_idx][0].category_name] = hand_gesture_recognition(fingers_extension[hand_idx])
    
    
    print(f'{hand_signs}')
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_results, hand_signs=hand_signs)

    return detection_results, annotated_image

def main():

    # STEP 1: Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='data/models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 2: Open Video Capture
    # define a video capture object
    cap = cv2.VideoCapture(0)

    # Main Loop
    while True:
        # capture image
        ret, frame = cap.read()

        if ret:
            detection_result, annotation = get_annotation_from(
                detector,
                frame
                )

            cv2.imshow('', annotation)
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
    main()