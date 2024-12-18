import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Variables
scroll_active = False
scroll_start_y = None
volume_active = False
last_copy_paste_time = 0
copy_paste_cooldown = 2
select_active = False

# Thresholds (adjust as needed)
scroll_threshold = 0.1
volume_threshold = 0.15
select_threshold = 0.1
scroll_stop_threshold = 0.05  # To stop scrolling when fingers are too close

# Helper functions
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

def fingers_up_count(landmarks):
    fingers_up = 0
    if landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y: fingers_up += 1
    if landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y: fingers_up += 1
    if landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y: fingers_up += 1
    if landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y: fingers_up += 1
    if landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y: fingers_up += 1
    return fingers_up

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
            landmarks = hand_landmarks.landmark
            # Extract landmarks for fingers
            index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            pinky_finger_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

            # Convert landmarks to screen coordinates
            index_x_px = int(index_finger_tip.x * frame.shape[1])
            index_y_px = int(index_finger_tip.y * frame.shape[0])
            thumb_x_px = int(thumb_tip.x * frame.shape[1])
            thumb_y_px = int(thumb_tip.y * frame.shape[0])
            middle_x_px = int(middle_finger_tip.x * frame.shape[1])
            middle_y_px = int(middle_finger_tip.y * frame.shape[0])
            pinky_x_px = int(pinky_finger_tip.x * frame.shape[1])
            pinky_y_px = int(pinky_finger_tip.y * frame.shape[0])

            # --- Mouse Control using Index Finger ---
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)
            pyautogui.moveTo(x, y)  # Move the mouse pointer to the index finger's position

            # --- Scrolling with 5 fingers up ---
            fingers_up = fingers_up_count(landmarks)

            if fingers_up == 5:  # If 5 fingers are up, scrolling is active
                scroll_diff = abs(index_finger_tip.y - middle_finger_tip.y)
                if scroll_diff > scroll_threshold:
                    if not scroll_active:
                        scroll_start_y = index_finger_tip.y
                        scroll_active = True
                    scroll_amount = int((scroll_start_y - index_finger_tip.y) * 100)
                    pyautogui.scroll(scroll_amount)
                    scroll_start_y = index_finger_tip.y
                else:
                    scroll_active = False
            else:
                scroll_active = False

            # --- Volume Control with thumb and pinky fingers ---
            volume_diff = abs(thumb_tip.y - pinky_finger_tip.y)
            if volume_diff < volume_threshold:
                if not volume_active:
                    volume_start_y = index_finger_tip.y
                    volume_active = True
                volume_change = int((volume_start_y - index_finger_tip.y) * 10)
                if volume_change > 0:
                    pyautogui.press('volumeup')
                elif volume_change < 0:
                    pyautogui.press('volumedown')
                volume_start_y = index_finger_tip.y
            else:
                volume_active = False

            # --- Select Text (Ctrl + A) - Left Hand: index + middle fingers ---
            select_distance = calculate_distance(index_finger_tip, middle_finger_tip)
            if select_distance < select_threshold:
                if not select_active:
                    select_active = True
                    pyautogui.hotkey('ctrl', 'a')  # Select all
            else:
                select_active = False

            # --- Copy Text (Ctrl + C) - Right Hand: index + middle fingers ---
            if calculate_distance(index_finger_tip, middle_finger_tip) < select_threshold:
                pyautogui.hotkey('ctrl', 'c')  # Copy

            # --- Paste Text (Ctrl + V) - Right Hand: thumb + pinky fingers ---
            if calculate_distance(thumb_tip, pinky_finger_tip) < select_threshold:
                pyautogui.hotkey('ctrl', 'v')  # Paste

            # Draw landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Mouse', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
