# Benchmarking script
# This script will parse All test images, their ground_truth class, perform our custom recognition
# And display evaluation results/metrics
#

# STEP 1: Import the necessary modules.
import os
import json
import cv2

from hand_sign_recognition.hand_detector import HandDetector
from hand_sign_recognition.hand_signs_recogniser import HandSignsRecogniser
from utils.landmarks_visualiser import draw_landmarks_on_image

def evaluation(test_run, classes=[]):
    """
        This function will calculate evaluation metrics for the classification task
        and return the evaluation results as a dict.
        Args:
            test_run (dict): Dictionary as following
          {
            'img0.png': {'gt': 'peace', 'pred': 'peace'},
            'img1.png': {'gt': 'surfer_sign', 'pred': 'open hand'},
            'img2.png': {'gt': 'peace', 'pred': 'peace'},
            'img3.png': {'gt': 'open hand', 'pred': 'open hand'}
            ...
          }
        
        Returns: Metrics for each class including TP, FP, FN, True Positives Rate (TPR) and F1-score.
            dict: 
          {
            'open_hand': {
              'TP': 0, 
              'FN': 1, 
              'FP': 0, 
              'TPR':TP/(TP+FN), #
              'f1-score': TP/(TP+0.5*(FP+FN))}
          }
    """
    # Extract all unique classes

    # Initialize metrics
    metrics = {cls: {'TP': 0, 'FP': 0, 'FN': 0} for cls in classes}

    # Calculate TP, FP, and FN for each class
    for img, result in test_run.items():
        print(f'Evaluation: {img} - {result}')
        gt = result['gt']  # Ground truth
        pred = result['pred']  # Prediction

        if gt == pred:
            metrics[gt]['TP'] += 1
        else:
            # False negative for the ground truth class
            metrics[gt]['FN'] += 1
            if pred in classes:
                # False positive for the predicted class
                metrics[pred]['FP'] += 1
            

    # Calculate TPR for each class
    for cls, values in metrics.items():
        TP = values['TP']
        FN = values['FN']
        FP = values['FP']
        metrics[cls]['TPR'] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        metrics[cls]['f1'] = TP / (TP+0.5*(TP + FN))

    return metrics

def print_evaluation_report(evaluation_metrics):
    for cls, values in evaluation_metrics.items():
        print(f"\n--Class: {cls}--\n----")
        print(f" | TP: {values['TP']} | FP: {values['FP']} | FN: {values['FN']} | TPR: {values['TPR']:.2f} | F1-Score: {values['f1']:.2f}\n--------------")

def main(test_set_path='data/test_data', model_asset_path="data/models/hand_landmarker.task", num_hands=1):

    # We use our HandDetector wrapper to intantiate the hands landmarks detector and
    # Configure it to detect one single hand.
    detector = HandDetector(model_asset_path=model_asset_path, num_hands=num_hands, stream=False)
    hand_sign_recogniser = HandSignsRecogniser()
    test_run_dict = {}
    # Parse test set and run inference per image.
    test_set_path = test_set_path
    classes = os.listdir(test_set_path)
    classes.sort()
    try:
        
        for gt_hand_sign in classes:
            image_files = os.listdir(os.path.join(test_set_path, gt_hand_sign))
            for file in image_files:
                if 'pred' not in file:
                    test_run_dict[file] = {}
                    test_run_dict[file]['gt'] = gt_hand_sign
                    image_path = os.path.join(test_set_path, gt_hand_sign, file)      
                    print(f'**** {image_path} ****')
                    # Hand gesture Recognition
                    print('Inferring image')
                    detection_result = detector.get_default_detections(filename=image_path)
                    hand_sign = hand_sign_recogniser.hand_gestures_recognition(detection_result)
                    print(f'Detected Hand sign for img {file} is {hand_sign}')
                    print('---')
                    # hand_sign is in format {'hand side': hand_sign}. We extract the value item
                    pred_hand_sign = list(hand_sign.values())[0]
                    test_run_dict[file]['pred'] = pred_hand_sign
                    image = cv2.imread(image_path)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    annotated_image = draw_landmarks_on_image(rgb_image, detection_result, hand_signs=hand_sign)
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(test_set_path, gt_hand_sign, f'preds_{file}'), annotated_image)
        print(test_run_dict)
    except Exception as e:
        print(e)

    evaluation_metrics = evaluation(test_run_dict, classes=classes)
    print_evaluation_report(evaluation_metrics)
    print('Done')


if __name__=="__main__":
    test_set_path='../data/test_data'
    model_asset_path="../data/models/hand_landmarker.task"
    num_hands=1
    main(
        test_set_path=test_set_path, 
        model_asset_path=model_asset_path, 
        num_hands=num_hands
    )