# GestureGo

GestureGo is a Python-based computer vision project that utilizes hand gestures to control system functions. It allows users to adjust system volume using their right hand and take screenshots using their left hand, all with simple and intuitive gestures.

---

## Features

### 1. Volume Control (Right Hand)
- **Gesture**: Move the **thumb** and **index finger** apart to increase the volume or bring them closer to decrease it.
- **Orientation**: The palm of the right hand should face the camera.
- **Range**: The system maps finger distance to the full volume range (0% to 100%).

### 2. Screenshot Capture (Left Hand)
- **Gesture**: Extend all five fingers of the left hand to take a screenshot.
- **Delay**: Screenshots are taken with a minimum delay of 3 seconds to prevent continuous captures.
- **Orientation**: The palm of the left hand should face the camera.

---

## Requirements

### Hardware
- A computer with a webcam.

### Software
1. Python 3.10 
2. Required Python libraries:
   - `opencv-python`
   - `mediapipe`
   - `numpy`
   - `pycaw`
   - `pyautogui`
   - `comtypes`

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/GestureGo.git
   cd GestureGo
## Run the script 
```bash
python GestureGo.py
