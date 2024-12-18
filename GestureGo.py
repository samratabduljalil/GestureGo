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
click_threshold = 0.3
scroll_threshold = 0.07  # Slightly increased for easier activation
scroll_speed_multiplier = 50 #Adjusted scroll speed

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

            # Convert landmarks to pixel coordinates for debugging
            index_x_px = int(index_finger_tip.x * frame.shape[1])
            index_y_px = int(index_finger_tip.y * frame.shape[0])
            pinky_x_px = int(pinky_finger_tip.x * frame.shape[1])
            pinky_y_px = int(pinky_finger_tip.y * frame.shape[0])
            middle_x_px = int(middle_finger_tip.x * frame.shape[1])
            middle_y_px = int(middle_finger_tip.y * frame.shape[0])

            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)
            pyautogui.moveTo(x, y)

            # Click Detection
            distance = calculate_distance(index_finger_tip, pinky_finger_tip)
            cv2.line(image, (index_x_px, index_y_px), (pinky_x_px, pinky_y_px), (0, 255, 0), 2) #Draw line
            cv2.putText(image, f"Click Dist: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #Show distance
            current_time = time.time()
            if distance < click_threshold and not clicking:
                pyautogui.click()
                clicking = True
                prev_time = current_time
                print("Click!") # Debug print
            elif distance >= click_threshold and clicking and (current_time - prev_time) > 0.5:
                clicking = False

            # Scroll Detection
            index_y = index_finger_tip.y
            middle_y = middle_finger_tip.y
            cv2.line(image, (index_x_px, index_y_px), (middle_x_px, middle_y_px), (255, 0, 0), 2) #Draw line
            cv2.putText(image, f"Scroll Diff: {abs(index_y - middle_y):.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) #Show diff
            if abs(index_y - middle_y) < scroll_threshold:
                if scroll_start_y is not None:
                    scroll_amount = int((scroll_start_y - index_y) * scroll_speed_multiplier)
                    pyautogui.scroll(scroll_amount)
                    print(f"Scrolling: {scroll_amount}") # Debug print
                scroll_start_y = index_y
            else:
                scroll_start_y = None

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Mouse', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()