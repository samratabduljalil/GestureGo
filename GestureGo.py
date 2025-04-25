import cv2
import numpy as np
import mediapipe as mp
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import pyautogui
import time

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Audio control initialization
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_vol, max_vol = volume_range[0], volume_range[1]

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Finger tip indices (Mediapipe Hand Landmark indices for fingertips)
finger_tips = [mp_hands.HandLandmark.THUMB_TIP,
               mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP]

finger_dips = [mp_hands.HandLandmark.THUMB_IP,
               mp_hands.HandLandmark.INDEX_FINGER_DIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
               mp_hands.HandLandmark.RING_FINGER_DIP,
               mp_hands.HandLandmark.PINKY_DIP]

last_screenshot_time = 0  # Timestamp of the last screenshot

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Check handedness
            handedness = result.multi_handedness[i].classification[0].label

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            h, w, _ = frame.shape
            landmarks = hand_landmarks.landmark

            # Detect palm orientation using wrist and middle finger MCP
            wrist = landmarks[mp_hands.HandLandmark.WRIST]
            middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            palm_orientation = wrist.z < middle_mcp.z  # True = palm facing camera; False = back side

            # Right Hand (Volume Control) - Only works with palm facing camera
            if handedness == "Right" and palm_orientation:
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_coords = (int(index_tip.x * w), int(index_tip.y * h))

                # Draw circles and line
                cv2.circle(frame, thumb_coords, 10, (255, 0, 0), -1)
                cv2.circle(frame, index_coords, 10, (255, 0, 0), -1)
                cv2.line(frame, thumb_coords, index_coords, (0, 255, 0), 3)

                # Calculate distance
                distance = hypot(index_coords[0] - thumb_coords[0], index_coords[1] - thumb_coords[1])

                # Adjusted range for smaller movements (distance range: 10 to 150)
                volume_level = np.interp(distance, [10, 150], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(volume_level, None)

                # Display volume level
                volume_bar = np.interp(distance, [10, 150], [400, 150])
                volume_percentage = np.interp(volume_level, [min_vol, max_vol], [0, 100])  # Volume as percentage
                cv2.rectangle(frame, (50, int(volume_bar)), (85, 400), (0, 255, 0), -1)
                cv2.putText(frame, f'Vol: {int(volume_percentage)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Left Hand (Screenshot) - Only works with palm facing camera
            elif handedness == "Left" and palm_orientation:
                fingers_extended = []
                for tip, dip in zip(finger_tips, finger_dips):
                    # Check if tip is above the dip for each finger
                    is_extended = landmarks[tip].y < landmarks[dip].y
                    fingers_extended.append(is_extended)

                # Screenshot Action (All five fingers extended)
                if all(fingers_extended):
                    current_time = time.time()
                    if current_time - last_screenshot_time > 3:  # 3-second delay
                        cv2.putText(frame, "Screenshot Taken", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        pyautogui.screenshot(f"screenshot_{int(current_time)}.png")
                        last_screenshot_time = current_time

    cv2.imshow("Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
