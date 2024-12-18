import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Variables for click and scroll
prev_time = 0
clicking = False
scroll_start_y = None
click_threshold = 0.1
scroll_threshold = 0.05 # Adjust for scroll sensitivity

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

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
            pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)
            pyautogui.moveTo(x, y)

            # Click Detection
            distance = calculate_distance(index_finger_tip, pinky_finger_tip)
            current_time = time.time()
            if distance < click_threshold and not clicking:
                pyautogui.click()
                clicking = True
                prev_time = current_time
            elif distance >= click_threshold and clicking and (current_time - prev_time) > 0.5:
                clicking = False

            # Scroll Detection
            index_y = index_finger_tip.y
            middle_y = middle_finger_tip.y

            if abs(index_y - middle_y) < scroll_threshold:  # Fingers close = scrolling
                if scroll_start_y is not None:
                    scroll_amount = int((scroll_start_y - index_y) * 100)  # Adjust multiplier for scroll speed
                    pyautogui.scroll(scroll_amount)
                scroll_start_y = index_y
            else:
                scroll_start_y = None  # Reset when fingers are not close

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Mouse', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()