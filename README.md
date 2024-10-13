
# Hand Gesture Recognition Using PyTorch and OpenCV

This project implements a **real-time hand gesture recognition system** using a deep learning model built with PyTorch and a webcam feed captured through OpenCV. The model can classify 10 different hand gestures and display the predicted gesture on the live video stream.

## Features
- **Real-time gesture detection**: The program captures video from your webcam and performs gesture classification on each frame.
- **Deep learning model**: The classification is performed using a convolutional neural network (CNN) trained to recognize 10 hand gestures.
- **Preprocessing**: Grayscale conversion and resizing of input frames for the model.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.10
- [PyTorch](https://pytorch.org/get-started/locally/)
- OpenCV (`opencv-python`)

You can install the required dependencies by running:
```bash
pip install torch torchvision opencv-python
```

### Clone the repository

```bash
git clone https://github.com/Hrishabh-V/PRODIGY_ML_04.git
cd PRODIGY_ML_04
```

### Model File

Make sure you have the pre-trained model file `gesture_recognition_model.pth`. Place it in the root directory of this repository. You can train your own model or download it if available.

## Dataset
- The dataset used for training the model can be found [here](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).


### Running the Application

To start real-time gesture recognition, run the following command:
```bash
python app.py
```

### This will open a webcam feed, and the recognized gesture will be displayed on the screen in real time.

## Code Overview

- **`app.py`**: Main script that loads the model, captures video, processes the frames, and performs gesture classification.
- **Gesture Recognition Model**: A simple Convolutional Neural Network (CNN) built using PyTorch.
  - Two convolutional layers followed by max pooling.
  - Two fully connected layers for classification.
  
### Gesture Labels

The following hand gestures are recognized by the model:

| Gesture ID | Gesture Name  |
|------------|---------------|
| 0          | Palm          |
| 1          | L             |
| 2          | Fist          |
| 3          | Fist Moved    |
| 4          | Thumb         |
| 5          | Index         |
| 6          | OK            |
| 7          | Palm Moved    |
| 8          | C             |
| 9          | Down          |

## Contributing

Feel free to open issues or submit pull requests if you find bugs or have suggestions for improvements!

