import tkinter as tk
from tkinter import messagebox
import numpy as np
import os
import time
import threading

from Get_bin import preberi_bin
from Obdelava_podatkov import sestavi_podatke

TOF = 5
MODEL_PATH = "model_tof.npz"


def extract_tof_feature(signal, fvz, target_len=128):
    if len(signal) < 10:
        return None

    x = signal.flatten().astype(np.float32)
    x = x - np.mean(x)

    fft = np.fft.rfft(x)
    mag = np.log1p(np.abs(fft))
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
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith(".bin"):
                continue

            file_path = os.path.join(folder, file)

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

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X = (X - X_mean) / X_std

    print("Dataset:", X.shape)
    print("Class balance:", np.bincount(Y.flatten()))

    return X, Y, X_mean, X_std


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

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, loss={loss:.4f}")

    def predict(self, x):
        _, out = self.forward(x)
        return np.argmax(out, axis=1)[0]

    # SAVE + MEAN/STD
    def save(self, path, mean, std):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 mean=mean,
                 std=std)

    def load(self, path):
        if not os.path.exists(path):
            return None, None

        data = np.load(path)

        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]

        return data["mean"], data["std"]


class App:
    def __init__(self):
        self.model = None
        self.X = None
        self.Y = None
        self.X_mean = None
        self.X_std = None

        self.root = tk.Tk()
        self.root.title("TOF REALTIME CLASSIFIER")

        tk.Button(self.root, text="Load Dataset", command=self.load_data).pack()
        tk.Button(self.root, text="Train", command=self.train).pack()
        tk.Button(self.root, text="Test", command=self.test).pack()
        tk.Button(self.root, text="Save", command=self.save).pack()
        tk.Button(self.root, text="Load Model", command=self.load).pack()
        tk.Button(self.root, text="Realtime", command=self.start_realtime).pack()

    def load_data(self):
        self.X, self.Y, self.X_mean, self.X_std = load_dataset()
        messagebox.showinfo("OK", f"Loaded {len(self.X)} samples")

    def train(self):
        if self.X is None:
            messagebox.showerror("Error", "Load dataset first")
            return

        self.model = NeuralNetwork(self.X.shape[1], hidden=32)
        self.model.train(self.X, self.Y, lr=0.01, epochs=200)

        messagebox.showinfo("Done", "Training finished")

    def test(self):
        if self.model is None:
            return

        i = np.random.randint(0, len(self.X))

        pred = self.model.predict(self.X[i].reshape(1, -1))
        true = self.Y[i][0]

        map_ = {0: "mirovanje", 1: "mahanje"}

        messagebox.showinfo(
            "Result",
            f"True: {map_[true]}\nPred: {map_[pred]}"
        )

    def save(self):
        if self.model:
            self.model.save(MODEL_PATH, self.X_mean, self.X_std)
            messagebox.showinfo("Saved", "Model saved")

    def load(self):
        if self.model is None:
            self.model = NeuralNetwork(128)

        mean, std = self.model.load(MODEL_PATH)

        if mean is None:
            messagebox.showerror("Error", "No model")
            return

        self.X_mean = mean
        self.X_std = std

        messagebox.showinfo("Loaded", "Model + scaler loaded")

    def predict_realtime(self, sig, fvz):
        feat = extract_tof_feature(sig, fvz)
        if feat is None:
            return None

        feat = (feat - self.X_mean) / (self.X_std + 1e-8)

        return self.model.predict(feat.reshape(1, -1))

    def realtime_loop(self, file_path):
        while True:
            paketi = preberi_bin(file_path)
            fvz, signals = sestavi_podatke(paketi)

            if TOF in signals and self.model is not None:
                sig = signals[TOF]

                pred = self.predict_realtime(sig, fvz[TOF])

                if pred is not None:
                    label = {0: "mirovanje", 1: "mahanje"}
                    print("REALTIME:", label[pred])

            time.sleep(0.5)

    def start_realtime(self):
        file_path = "tof_meritve/live.bin"

        t = threading.Thread(
            target=self.realtime_loop,
            args=(file_path,),
            daemon=True
        )
        t.start()

        messagebox.showinfo("OK", "Realtime started")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()

