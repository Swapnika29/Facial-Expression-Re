Project Name : Facial-Expression-Recognition

Topic : Development of an Efficient Deep Learning Model for Real-Time Facial Emotion Recognition

Data Set Source 

https://www.kaggle.com/datasets/prajwalsood/google-fer-image-format/data

Group Members 

1.) Mounika Teppola - 121012135
2.) Bala Swapnika Gopi - 121332264
3.) Harshitha Gaddambachahalli Raghukumar - 1211162144
4.) Saee Desai - 121206442

# Facial Emotion Recognition using MobileNet

## Overview
This project implements a facial emotion recognition system using a pre-trained MobileNet model. The application detects facial expressions in real-time via webcam and classifies them into one of eight categories: Angry, Contempt, Disgust, Fear, Happy, Sad, Surprise, or Neutral.

## Setup Instructions

1. **Create a virtual environment:**
    ```bash
    conda create -n facial python=3.10 -y
    conda activate facial
    ```

2. **Install dependencies:**
    ```bash
    pip install tensorflow opencv-python
    ```

3. **Run the application:**
    ```bash
    python main.py
    ```

## Files

### `main.py`
This script runs the real-time facial emotion recognition application. It uses OpenCV to capture video frames from the webcam and detects faces using a Haar cascade classifier. Detected faces are preprocessed and passed to the MobileNet model for emotion prediction.

### `mobilenet.ipynb`
This Jupyter Notebook contains the training and evaluation code for the MobileNet-based emotion recognition model. It details data preprocessing, model training, and saving the trained model as `20_epochs_mobilenet_reloaded.keras`.

## Model
- The pre-trained MobileNet model is saved as `20_epochs_mobilenet_reloaded.keras`.
- Ensure the model file is placed in the same directory as `main.py` or update the file path in the script accordingly.

## Usage Notes
- Press `ctrl + q` to quit the application during runtime.
- Ensure your webcam is connected and accessible.

## Requirements
- Python 3.10
- TensorFlow
- OpenCV

## Troubleshooting
- **No webcam feed:**
  - Ensure your webcam is properly connected and accessible.

- **Missing model file:**
  - Verify that `20_epochs_mobilenet_reloaded.keras` exists in the correct directory.

- **Errors related to OpenCV or TensorFlow:**
  - Reinstall the dependencies using the command: `pip install tensorflow opencv-python`.

                





