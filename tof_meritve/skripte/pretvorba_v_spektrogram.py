import os
import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt

from obdelava_signalov import preprocess_signal

FS = 50

NPERSEG = 64
NOVERLAP = 48
NFFT = 128


def split_signal(x, fs, window_sec=2.0, overlap=0.5):
    step = int(window_sec * fs * (1 - overlap))
    size = int(window_sec * fs)
    segments = []
    for start in range(0, len(x) - size + 1, step):
        segments.append(x[start:start + size])
    return segments

def to_spectrogram(x, fs):
    f, t, Sxx = signal.spectrogram(
        x,
        fs=fs,
        window='hamming',
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        scaling='spectrum',
        mode='magnitude'
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    vmin = np.percentile(Sxx_db, 5)
    vmax = np.percentile(Sxx_db, 95)
    Sxx_db = np.clip(Sxx_db, vmin, vmax)
    Sxx_norm = (Sxx_db - vmin) / (vmax - vmin + 1e-10)
    img = (Sxx_norm * 255).astype(np.uint8)
    return img, f, t, Sxx_db

def process_file(filepath, output_dir, fs):
    x = np.fromfile(filepath, dtype=np.int16).astype(float)
    x = preprocess_signal(x, fs)
    segments = split_signal(x, fs)
    for i, seg in enumerate(segments):
        img, f, t, _ = to_spectrogram(seg, fs)
        filename = os.path.splitext(os.path.basename(filepath))[0]
        out_name = f"{filename}_{i}.png"
        out_path = os.path.join(output_dir, out_name)
        Image.fromarray(img).save(out_path)
    return x, f, t