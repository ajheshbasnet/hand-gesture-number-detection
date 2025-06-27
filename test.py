import cv2
import pickle
import mediapipe as mp

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)



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

cap = cv2.VideoCapture(0)


while True:

    _, frame = cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    landmarks = hand_pose(frame)
    if landmarks:
        y_pred = model.predict([landmarks])
        y_pred +=1

        x, y, _ = frame.shape
        cv2.putText(frame, f"The number is {y_pred}", (50, 50) , cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Frames", frame)
    else:
        print("No frames were detected")
    if cv2.waitKey(4) & 0xFF == ord('q'):
        break

cap.release()
