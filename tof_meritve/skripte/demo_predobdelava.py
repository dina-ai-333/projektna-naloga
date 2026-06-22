import os
import numpy as np
import matplotlib.pyplot as plt

from obdelava_signalov import preprocess_signal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FS = 50


def load_bin(path):
    return np.fromfile(path, dtype=np.int16).astype(float)


def demo(filepath, label):

    raw = load_bin(filepath)

    zero_mask = (raw == 0)

    clean = preprocess_signal(raw, FS)

    fig, axs = plt.subplots(2, 1, figsize=(14, 7))

    axs[0].plot(raw, color='tab:blue')

    axs[0].scatter(
        np.where(zero_mask)[0],
        raw[zero_mask],
        color='red',
        s=30,
        label='Nicelne vrednosti'
    )

    axs[0].set_title("Originalni signal")
    axs[0].set_xlabel("Vzorec")
    axs[0].set_ylabel("Amplituda")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(clean, color='green')

    axs[1].set_title("Signal po predobdelavi")
    axs[1].set_xlabel("Vzorec")
    axs[1].set_ylabel("Normalizirana amplituda")
    axs[1].grid(True)

    plt.suptitle(label)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    demo(
        os.path.join(BASE_DIR, "dataset_bin/mahanje/log012.bin"),
        "MAHANJE"
    )

    demo(
        os.path.join(BASE_DIR, "dataset_bin/mirovanje/log111.bin"),
        "MIROVANJE"
    )