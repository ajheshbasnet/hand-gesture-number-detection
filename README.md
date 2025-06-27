# Hand Gesture Number Detection

This is a simple computer vision project where I built a **hand gesture-based number recognition system**.  
The system can detect hand signs and predict the number being shown.

## ✋ Project Description
- I created my own **synthetic dataset** by recording videos of hand gestures showing different numbers.
- I converted the videos into image sequences using an **online video-to-image conversion tool**.
- I trained a machine learning model using these hand gesture frames.
- The system uses **MediaPipe** for hand landmark detection and **OpenCV** for real-time video processing.

## 🚀 Features
- Real-time hand gesture detection.
- Number prediction based on hand poses.
- Fully self-made dataset without using pre-existing datasets.

## 📂 Project Highlights
- 📹 **Dataset Creation:**  
  I recorded videos showing proper hand number symbols (0, 1, 2, ...).

- 🖼️ **Frame Extraction:**  
  Used online tools to convert video to image sequences for dataset generation.

- 🖐️ **Model Training:**  
  Trained a model using extracted hand landmarks to classify hand signs into corresponding numbers.

## 🔧 Tech Stack
- Python
- OpenCV
- MediaPipe
- Scikit-Learn (Random Forest Classifier)

## 💻 How to Run
```bash
python test.py
