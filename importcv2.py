import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import serial
import time

class MultiLabelPathClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 3)  # blocked / unblocked / yellow

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = path_classifier().to(device)
model.load_state_dict(torch.load("path_classifier.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

labels = ["Blocked Path", "Unblocked Path", "Yellow Path"]
threshold = 0.5

try:
    ser = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)
except serial.SerialException as e:
    ser = None
    print("Serial error:", e)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0)

    detected_labels = [labels[i] for i, p in enumerate(probs) if p > threshold]

    if ser:
        blocked = "Blocked Path" in detected_labels
        unblocked = "Unblocked Path" in detected_labels
        yellow = "Yellow Path" in detected_labels

        if blocked:
            ser.write(b'B')   # blocked
            print("CMD: BLOCKED (B)")
        elif unblocked and yellow:
            ser.write(b'G')   # good to go
            print("CMD: GO (G)")
        elif not yellow:
            ser.write(b'N')   # no yellow
            print("CMD: NO YELLOW (N)")

    display_text = ", ".join(detected_labels) if detected_labels else "None"
    cv2.putText(frame, f"Detected: {display_text}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Path Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()

