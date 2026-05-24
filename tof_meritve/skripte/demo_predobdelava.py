import os
import numpy as np
import matplotlib.pyplot as plt
from obdelava_signalov import preprocess_signal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FS = 50

def load_bin(path):
    return np.fromfile(path, dtype=np.int16).astype(float)

def demo(filepath, label):
    x = load_bin(filepath)
    raw = x.copy()
    clean = preprocess_signal(x, FS)

    print("RAW:", raw[:10])
    print("CLEAN:", clean[:10])

    print("RAW min/max:", raw.min(), raw.max())
    print("CLEAN min/max:", clean.min(), clean.max())

    plt.figure(figsize=(12,4))
    plt.plot(clean / np.max(np.abs(clean)), label="Obdelan (normiran)")
    plt.plot(raw / np.max(np.abs(raw)), alpha=0.4, label="Surov (normiran)")
    plt.title(f"{label} - {os.path.basename(filepath)}")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    demo(os.path.join(BASE_DIR, "dataset_bin/mahanje/log012.bin"), "MAHANJE")
    demo(os.path.join(BASE_DIR, "dataset_bin/mahanje/log015.bin"), "MAHANJE")
    demo(os.path.join(BASE_DIR, "dataset_bin/mirovanje/log111.bin"), "MIROVANJE")