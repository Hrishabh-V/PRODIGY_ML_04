import cv2
import torch
import torch.nn as nn  
import numpy as np
from torchvision import transforms

# Define the gesture labels first
gesture_labels = {
    0: 'palm',
    1: 'l',
    2: 'fist',
    3: 'fist_moved',
    4: 'thumb',
    5: 'index',
    6: 'ok',
    7: 'palm_moved',
    8: 'c',
    9: 'down'
}

# Load the trained model
class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, len(gesture_labels))  
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = GestureRecognitionModel()
model.load_state_dict(torch.load('gesture_recognition_model.pth'))
model.eval()

# Preprocessing function
def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    gray = cv2.resize(gray, (128, 128))  
    gray = gray / 255.0  # Normalize
    gray = torch.tensor(gray, dtype=torch.float32).unsqueeze(0)  
    gray = gray.unsqueeze(0)  
    return gray

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess_image(frame)

    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        gesture = gesture_labels[predicted.item()]

    # Display the gesture on the frame
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
