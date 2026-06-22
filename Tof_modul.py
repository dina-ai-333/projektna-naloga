import tkinter as tk
from tkinter import messagebox
import numpy as np
import os
import time

from Get_bin import preberi_bin
from Obdelava_podatkov import sestavi_podatke

TOF = 5
MODEL_PATH = "model_tof.npz"

# Nalaganje TOF
def extract_tof_feature(signal, fvz, target_len=128):
    if len(signal) < 10:
        return None

    x = signal.flatten().astype(np.float32)
    x = x - np.mean(x)

    fft = np.fft.rfft(x)

    # LOG FFT 
    mag = np.log1p(np.abs(fft))

    # normalizacija
    mag = mag / (np.linalg.norm(mag) + 1e-8)

    if len(mag) < target_len:
        mag = np.pad(mag, (0, target_len - len(mag)))
    else:
        mag = mag[:target_len]

    return mag


def load_dataset(base_path="tof_meritve"):
    X, Y = [], []

    dataset_path = os.path.join(base_path, "dataset_bin")

    classes = {
        "mirovanje": 0,
        "mahanje": 1
    }

    print("Dataset folders:", os.listdir(dataset_path))

    for class_name, label in classes.items():
        folder = os.path.join(dataset_path, class_name)

        if not os.path.exists(folder):
            print("Missing folder:", folder)
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith(".bin"):
                continue

            file_path = os.path.join(folder, file)
            print("Loading:", file_path)

            paketi = preberi_bin(file_path)
            fvz, signals = sestavi_podatke(paketi)

            if TOF not in signals:
                continue

            sig = signals[TOF]
            feat = extract_tof_feature(sig, fvz[TOF])

            if feat is None:
                continue

            X.append(feat)
            Y.append(label)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32).reshape(-1, 1)

    # Standardizacija
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X = (X - X_mean) / X_std

    print("FINAL dataset:", X.shape)
    print("class balance:", np.bincount(Y.flatten()))

    return X, Y, X_mean, X_std



# Nevronska mreža
class NeuralNetwork:
    def __init__(self, input_size, hidden=32):
        self.W1 = np.random.randn(input_size, hidden) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden))

        self.W2 = np.random.randn(hidden, 2) * np.sqrt(2 / hidden)
        self.b2 = np.zeros((1, 2))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        e = np.clip(e, 1e-8, 1e8)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.softmax(z2)

        return a1, a2

    def train(self, X, Y, lr=0.01, epochs=200):

        Y = Y.flatten()

        for epoch in range(epochs):

            z1 = X @ self.W1 + self.b1
            a1 = self.relu(z1)

            z2 = a1 @ self.W2 + self.b2
            a2 = self.softmax(z2)
            
            Y_oh = np.zeros((len(Y), 2))
            Y_oh[np.arange(len(Y)), Y] = 1

            loss = -np.mean(np.sum(Y_oh * np.log(a2 + 1e-8), axis=1))

            dz2 = a2 - Y_oh

            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0)

            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, loss={loss:.4f}")

    def predict(self, x):
        _, out = self.forward(x)
        return np.argmax(out, axis=1)[0]

    def save(self, path=MODEL_PATH):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2)

    def load(self, path=MODEL_PATH):
        if not os.path.exists(path):
            return False

        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        return True

class App:
    def __init__(self):
        self.model = None
        self.X = None
        self.Y = None
        self.X_mean = None
        self.X_std = None

        self.root = tk.Tk()
        self.root.title("TOF Classifier FIXED")

        tk.Button(self.root, text="Load Dataset", command=self.load_data).pack()
        tk.Button(self.root, text="Train", command=self.train).pack()
        tk.Button(self.root, text="Test Sample", command=self.test).pack()
        tk.Button(self.root, text="Save", command=self.save).pack()
        tk.Button(self.root, text="Load", command=self.load).pack()

    def load_data(self):
        self.X, self.Y, self.X_mean, self.X_std = load_dataset()
        messagebox.showinfo("OK", f"Loaded {len(self.X)} samples")

    def train(self):
        if self.X is None:
            messagebox.showerror("Error", "Load dataset first")
            return

        self.model = NeuralNetwork(self.X.shape[1], hidden=32)

        start = time.time()
        self.model.train(self.X, self.Y, lr=0.01, epochs=200)

        print("Training time:", time.time() - start)
        messagebox.showinfo("Done", "Training finished")

    def test(self):
        if self.model is None:
            messagebox.showerror("Error", "Train model first")
            return

        i = np.random.randint(0, len(self.X))

        pred = self.model.predict(self.X[i].reshape(1, -1))

        true = self.Y[i][0]

        label_map = {0: "mirovanje", 1: "mahanje"}

        print("PRED:", pred, "TRUE:", true)

        messagebox.showinfo(
            "Result",
            f"True: {label_map[true]}\nPred: {label_map[pred]}"
        )

    def save(self):
        if self.model:
            self.model.save()
            messagebox.showinfo("Saved", "Model saved")

    def load(self):
        if self.X is None:
            return

        self.model = NeuralNetwork(self.X.shape[1])
        if self.model.load():
            messagebox.showinfo("Loaded", "Model loaded")
        else:
            messagebox.showerror("Error", "No model found")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()