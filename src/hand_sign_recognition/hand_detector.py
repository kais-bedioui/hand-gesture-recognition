import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class HandDetector():
    """
       This is a wrapper around mediapipe's hand landmarks detector.
       It can initiate a hand_landmarks detector and handle inference and getting results
    """

    def __init__(self, model_asset_path='data/models/hand_landmarker.task', num_hands=2, stream=True):
        # STEP 1: Create an HandLandmarker object.
        """
        Intantiate a Hands Landmarks detector
        Args:
            model_asset_path (string): path to model .task file
            num_hands (int): Number of hands we expect in the image
            stream (bool): Set to true if it's a webcam input, False if to read from image
        """
        base_options = python.BaseOptions(
            model_asset_path=model_asset_path
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.stream = stream

    def get_default_detections(self, cv2_frame=None, filename=""):
        """
            This inference function will extract Hand Landmarks and Handedness from provided image.
            image: CV2 np array
        """
        if not self.stream:
            frame = cv2.imread(filename)
            # The next conversion is necessary to successfully run the detection
            # and to avoid this error message: 
            # W0000 00:00:1731884788.399095   96173 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
            # Fix per: https://stackoverflow.com/a/76280935
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2_frame = np.asarray(frameRGB)
            
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_frame)
        detection_results = self.detector.detect(image)
        return detection_results
    
    def get_json_detections(self, cv2_frame=None, filename=""):
        """
           This inference function will extract Hand Landmarks and Handedness from provided image
           in JSON format {'hand_landmarks': hand_landmarks, 'handedness': handedness}
           
           image: CV2 np array
        """
        json_detections = {}
        if not self.stream:
            frame = cv2.imread(filename)
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2_frame = np.asarray(frameRGB)
            
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_frame)
        detection_results = self.detector.detect(image)
        hand_landmarks = detection_results.hand_landmarks
        handedness = detection_results.handedness
        json_detections['hand_landmarks'] = hand_landmarks
        json_detections['handedness'] = handedness
        return json_detections
    
if __name__=="__main__":
    detector = HandDetector()
    