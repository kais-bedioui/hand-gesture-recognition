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

    def __init__(self, model_asset_path='data/models/hand_landmarker.task', num_hands=2):
        # STEP 1: Create an HandLandmarker object.
        base_options = python.BaseOptions(
            model_asset_path=model_asset_path
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def get_default_detections(self, image):
        """
            This inference function will extract Hand Landmarks and Handedness from provided image
        """
        detection_results = self.detector.detect(image)
        return detection_results
    
    def get_json_detections(self, image):
        """
           This inference function will extract Hand Landmarks and Handedness from provided image
           in JSON format {'hand_landmarks': hand_landmarks, 'handedness': handedness}
        """
        json_detections = {}
        detection_results = self.detector.detect(image)
        hand_landmarks = detection_results.hand_landmarks
        handedness = detection_results.handedness
        json_detections['hand_landmarks'] = hand_landmarks
        json_detections['handedness'] = handedness
        return json_detections
    
if __name__=="__main__":
    detector = HandDetector()
    