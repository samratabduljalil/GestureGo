# GestureGo: Hand Mouse Using Machine Learning

**GestureGo** allows users to control their computer's mouse using hand gestures, utilizing machine learning for hand tracking and gesture recognition. This project uses the `MediaPipe` library for hand gesture detection and `PyAutoGUI` for mouse control.

## Features
- **Real-time Hand Tracking**: Tracks the movement of your hand using the webcam.
- **Gesture Control**: Move the mouse cursor with the movement of your hand, particularly using the index finger.
- **No Hardware Required**: Uses only a webcam and your hand gestures.

## Requirements

- Python 3.6+
- Libraries:
  - `mediapipe`: For hand gesture recognition and landmark detection.
  - `opencv-python`: For capturing video frames.
  - `pyautogui`: For controlling the mouse cursor.

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/GestureGo.git
    cd GestureGo
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, you can manually install the dependencies:

    ```bash
    pip install mediapipe opencv-python pyautogui
    ```

## Usage

1. Run the Python script:

    ```bash
    python hand_mouse.py
    ```

2. The webcam will activate, and you can control the mouse cursor with your hand gestures.
3. To stop the program, press `q` on your keyboard.

## Code Explanation

- **Hand Tracking**: Uses `MediaPipe` to detect the hand landmarks in real-time.
- **Mouse Movement**: Maps the location of the index finger's tip (landmark 8) to the screen coordinates using `PyAutoGUI` to move the mouse cursor.
- **Webcam Capture**: OpenCV captures frames from the webcam, which are processed for hand landmarks.

## Contributing

Feel free to fork this repository, contribute to the project, or submit issues and pull requests. If you have any improvements or suggestions, please open an issue or send a pull request!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MediaPipe**: A framework developed by Google for building multimodal applied machine learning pipelines.
- **PyAutoGUI**: A Python library used for GUI automation, including controlling the mouse.
- **OpenCV**: A popular computer vision library used to capture and process video frames.

## Contact

If you have any questions, feel free to reach out to me:

- Email: samrat@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

