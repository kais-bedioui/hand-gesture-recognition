import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify

from hand_sign_recognition.hand_detector import HandDetector
from hand_sign_recognition.hand_signs_recogniser import HandSignsRecogniser

# Initialize Flask app
app = Flask(__name__)
hand_detector = HandDetector(
    model_asset_path="../data/models/hand_landmarker.task",
    num_hands=2,
    stream=True, #inferencing on a single frame
)
hand_sign_recogniser = HandSignsRecogniser()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Parse JSON input
        data = request.json
        encoded_image = data.get("image")

        if not encoded_image:
            return jsonify({"error": "No image provided"}), 400

        # Step 2: Decode Base64 image
        decoded_image = base64.b64decode(encoded_image)
        np_image = np.frombuffer(decoded_image, np.uint8)
        # Per https://stackoverflow.com/a/76280935
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)[..., ::-1].copy()

        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        # Step 3: Recognize hand sign
        hand_landmarker_results = hand_detector.get_hand_landmarks(cv2_frame=frame[..., ::-1].copy())
        detected_signs = hand_sign_recogniser.hand_gestures_recognition(hand_landmarker_results)

        # Step 4: Return result
        return jsonify({"predicted_signs": detected_signs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)