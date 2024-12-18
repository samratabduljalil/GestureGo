import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Variables for click and scroll
prev_time = 0
clicking = False
scroll_start_y = None
click_threshold = 0.1  # Adjust as needed
scroll_threshold = 0.07
scroll_speed_multiplier = 50

# Smoothing parameters
smoothing_window = 3
index_x_history = deque(maxlen=smoothing_window)
index_y_history = deque(maxlen=smoothing_window)

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

def smooth_coordinates(x, y):
    index_x_history.append(x)
    index_y_history.append(y)
    smoothed_x = np.mean(index_x_history)
    smoothed_y = np.mean(index_y_history)
    return smoothed_x, smoothed_y

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]  # For click
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP] #For Scroll

            index_x_px = int(index_finger_tip.x * frame.shape[1])
            index_y_px = int(index_finger_tip.y * frame.shape[0])
            thumb_x_px = int(thumb_tip.x * frame.shape[1])
            thumb_y_px = int(thumb_tip.y * frame.shape[0])
            middle_x_px = int(middle_finger_tip.x * frame.shape[1])
            middle_y_px = int(middle_finger_tip.y * frame.shape[0])

            # Smoothing
            smoothed_x, smoothed_y = smooth_coordinates(index_finger_tip.x, index_finger_tip.y)
            x = int(smoothed_x * screen_width)
            y = int(smoothed_y * screen_height)
            pyautogui.moveTo(x, y)

            # Click Detection (Thumb and Index)
            click_distance = calculate_distance(index_finger_tip, thumb_tip)
            cv2.line(image, (index_x_px, index_y_px), (thumb_x_px, thumb_y_px), (0, 255, 0), 2)  # Green line
            cv2.putText(image, f"Click Dist: {click_distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            current_time = time.time()
            if click_distance < click_threshold and not clicking:
                pyautogui.click()
                clicking = True
                prev_time = current_time
                print("Click!")
            elif click_distance >= click_threshold and clicking and (current_time - prev_time) > 0.5:
                clicking = False

            # Scroll Detection (Index and Middle)
            scroll_diff = abs(index_finger_tip.y - middle_finger_tip.y)
            cv2.line(image, (index_x_px, index_y_px), (middle_x_px, middle_y_px), (255, 0, 0), 2)  # Blue line
            cv2.putText(image, f"Scroll Diff: {scroll_diff:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if scroll_diff < scroll_threshold:
                if scroll_start_y is not None:
                    scroll_amount = int((scroll_start_y - index_finger_tip.y) * scroll_speed_multiplier)
                    pyautogui.scroll(scroll_amount)
                    print(f"Scrolling: {scroll_amount}")
                scroll_start_y = index_finger_tip.y
            else:
                scroll_start_y = None

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Mouse', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()