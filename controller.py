import numpy as np

from tof_detector import TOFDetector
from gesture_detector import detect_gesture

MODEL_PATH = "model_tof.npz"
COM_PORT = "COM3"


class NeuralNetwork:

    def __init__(self, input_size=128, hidden=32):

        self.W1 = None
        self.b1 = None

        self.W2 = None
        self.b2 = None

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):

        x = x - np.max(x)

        e = np.exp(x)

        return e / np.sum(e)

    def predict(self, x):

        z1 = x @ self.W1 + self.b1

        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2

        probs = self.softmax(z2)

        return np.argmax(probs)

    def load(self, path):

        data = np.load(path)

        self.W1 = data["W1"]
        self.b1 = data["b1"]

        self.W2 = data["W2"]
        self.b2 = data["b2"]

        mean = data["mean"]
        std = data["std"]

        return mean, std


model = NeuralNetwork()

mean, std = model.load(MODEL_PATH)

tof = TOFDetector(
    model,
    mean,
    std,
    port=COM_PORT
)

while True:

    print()
    print("Waiting for wave...")

    tof.wait_for_wave()

    print("Wave detected!")

    gesture = detect_gesture(timeout=30)

    if gesture:

        print()
        print("Detected gesture:")
        print(gesture)

    else:

        print()
        print("No gesture detected")