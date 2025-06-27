import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

def hand_pose(img):

    mp_hands = mp.solutions.hands

    landmarks = []

    with mp_hands.Hands(static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.5) as hands:

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for idx, points in enumerate(results.multi_hand_landmarks[0].landmark):
                landmarks.append(points.x)
                landmarks.append(points.y)
                landmarks.append(points.z)
            
    return landmarks        


dataset = r'C:\Users\hp\OneDrive\Desktop\computer vision\hand-pose\Hands Data'

final_dataset = []
for pose_idx, num_finger in enumerate(sorted(os.listdir(dataset))):
    
    poses_path = os.path.join(dataset, num_finger)

    for img_path in os.listdir(poses_path):

        img_path = os.path.join(poses_path, img_path)
        img = cv2.imread(img_path)
        poses =  hand_pose(img)

        if len(poses) == 63:
            poses.append(pose_idx)
            final_dataset.append(poses)


df = pd.DataFrame(final_dataset)
df.to_csv('hand-poses.csv', index=False)
print("Dataset successfully saved!!")