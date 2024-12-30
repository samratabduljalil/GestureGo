import cv2
import numpy as np
import mediapipe as mp
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Audio control initialization
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_vol, max_vol = volume_range[0], volume_range[1]

# OpenCV video capture
cap = cv2.VideoCapture(0)

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
            # Check handedness (Right hand)
            handedness = result.multi_handedness[i].classification[0].label
            if handedness != "Right":
                continue

            # Draw landmarks on the right hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions for thumb and index finger tips
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            h, w, _ = frame.shape
            thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_coords = (int(index_tip.x * w), int(index_tip.y * h))

            # Draw circles and line
            cv2.circle(frame, thumb_coords, 10, (255, 0, 0), -1)
            cv2.circle(frame, index_coords, 10, (255, 0, 0), -1)
            cv2.line(frame, thumb_coords, index_coords, (0, 255, 0), 3)

            # Calculate distance
            distance = hypot(index_coords[0] - thumb_coords[0], index_coords[1] - thumb_coords[1])

            # Map distance to volume range
            volume_level = np.interp(distance, [30, 300], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(volume_level, None)

            # Display volume level
            volume_bar = np.interp(distance, [30, 300], [400, 150])
            cv2.rectangle(frame, (50, int(volume_bar)), (85, 400), (0, 255, 0), -1)
            cv2.putText(frame, f'Vol: {int(volume_level)}', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
