import math
import numpy as np

HAND_SIGNS = {
    "open_hand": [True, True, True, True, True],
    "peace_sign": [False, True, True, False, False],
    "surfer_sign": [True, False, False, False, True],
    "three_pointer": [False, False, True, True, True]
}

class HandSignsRecogniser():
    def __init__(self, hand_signs=HAND_SIGNS):
        """
            Intantiate object with finger orders and names, hands orders, and
            hand signs to recognized.
            Each hand sign is a boolean vector where each element of index i
            represents if finger i is extended(True) or folded(False).
            Example:
            if fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"] then 
            Peace_sign = [ False,   True,    True,    False,   False]
        """

        self.fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.hands = ["Right", "Left"]
        # Define
        self.hand_signs = hand_signs

    def hand_gestures_recognition(self, detection_results):
        """
            This is the main classification function. We pass as arguments the
            output from the HandDetector object, which contains handedness and hand landmarks,
            and it will output the hand sign.
        """
        hand_signs = {}
        fingers_extension = []
        handedness = detection_results.handedness ## [[CategoryName], [CategoryName]]
        hand_landmarks = detection_results.hand_landmarks 

        for hand_idx in range(len(hand_landmarks)):
            hand_signs[handedness[hand_idx][0].category_name] = ""
            fingers_extension.append([])

            current_hand_wrist_landmarks = self._extract_wrist_landmarks(
                hand_landmarks=hand_landmarks[hand_idx]
            )

            for idx, finger in enumerate(self.fingers):
                
                finger_index = idx + 1
                finger_landmarks = self._extract_finger_landmarks(
                    hand_landmarks=hand_landmarks[hand_idx], 
                    finger_index=finger_index, 
                    finger_name=finger
                )
                
                if finger_landmarks and current_hand_wrist_landmarks:
                    is_extended = self._finger_is_extended(
                        finger_landmarks=finger_landmarks, 
                        wrist_landmarks=current_hand_wrist_landmarks,
                        finger=finger
                )
                
                fingers_extension[hand_idx].append(is_extended)

            fingers_extension_vector = np.array(fingers_extension[hand_idx])

            for sign, one_hot_vector in self.hand_signs.items():
                if np.array_equal(fingers_extension_vector, one_hot_vector):
                    hand_signs[handedness[hand_idx][0].category_name] = sign
                    break
        
        return hand_signs

    def _extract_finger_landmarks(self, hand_landmarks=[], finger_index=0, finger_name="Finger"):
        finger_mcp_index = 4*finger_index - 3
        finger_tip_index = 4*finger_index
        finger_landmarks = hand_landmarks[finger_mcp_index:finger_tip_index+1]
        return finger_landmarks

    def _extract_wrist_landmarks(self, hand_landmarks=[]):

        if hand_landmarks:
            wrist_landmarks = hand_landmarks[0]
            return wrist_landmarks
        else:
            return []
    
    def _finger_is_extended(self, finger_landmarks=[], wrist_landmarks=[], finger="Finger"):
        """
            This function will measure and compare the distances d1=(wrist-finger MCP) & d2=(wrist-finger TIP)
            If d1 > d2: finger folded
            if d2 > d1: finger extended
        """

        wrist_x = wrist_landmarks.x
        wrist_y = wrist_landmarks.y

        finger_mcp_x = finger_landmarks[0].x
        finger_mcp_y = finger_landmarks[0].y
        finger_pip_x = finger_landmarks[1].x
        finger_pip_y = finger_landmarks[1].y
        finger_tip_x = finger_landmarks[3].x
        finger_tip_y = finger_landmarks[3].y
        
        if finger.lower() == "thumb":
            if wrist_x > finger_mcp_x:
                extended = finger_tip_x < finger_pip_x
            elif wrist_x < finger_mcp_x:
                extended = finger_tip_x > finger_pip_x
            else:
                extended = False

            #dist_wrist_to_mcp = abs(wrist_x - finger_mcp_x)
            #dist_wrist_to_tip = abs(wrist_x - finger_tip_x)
            #extended = dist_wrist_to_tip > dist_wrist_to_mcp
        else:
            dist_wrist_to_pip = math.dist(
                (wrist_x, wrist_y),
                (finger_pip_x, finger_pip_y)
            )
            dist_wrist_to_tip = math.dist(
                (wrist_x, wrist_y),
                (finger_tip_x, finger_tip_y)
            )
        
            extended = dist_wrist_to_tip > dist_wrist_to_pip
        return extended