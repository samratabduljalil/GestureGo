# Hand Gesture Control for Mouse and Keyboard

This project allows you to control the mouse and perform various keyboard actions using hand gestures captured through your webcam. Using **Mediapipe** for hand tracking and **PyAutoGUI** for controlling the mouse and keyboard, this project enables hands-free control of your computer.

## Features

- **Mouse Control**: Move the mouse cursor using the **index finger**.
- **Scrolling**: Scroll up or down using **five fingers raised**.
- **Volume Control**: Adjust the volume using the **thumb** and **pinky** fingers.
- **Text Selection**: Select all text using **left hand's index and middle fingers** (Ctrl + A).
- **Copy**: Copy selected text using **right hand's index and middle fingers** (Ctrl + C).
- **Paste**: Paste text using **right hand's thumb and pinky fingers** (Ctrl + V).

## Requirements

To run this project, you need:

- Python 3.6 or higher
- Webcam for capturing hand gestures
- **Mediapipe**: For hand gesture recognition
- **OpenCV**: For video capture and image processing
- **PyAutoGUI**: For controlling the mouse and keyboard

## Installation

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/hand-gesture-control.git
cd hand-gesture-control
