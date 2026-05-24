import os
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter, butter, filtfilt
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FS = 500 #vzorčevalna frekvenca

WINDOW_SEC = 2.0 #ker dogodek mahanja traja približno toliko
OVERLAP = 0.5 #bolj gradek prehod med segmenti

NPERSEG = 64 #kompromis med čas in frek ločljivostjo
NOVERLAP = 48 #naslednje okno se prekriva za 48 vzorcev
NFFT = 128 #bolj fina frekvenčna os

#nizko prepustno sito
def lowpass_filter(x, fs, cutoff=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='low')
    return filtfilt(b, a, x)

def preprocess_signal(x, fs):
    x = x.copy()
    x[x == 0] = np.nan
    mask = np.isnan(x)
    if np.any(mask):
        x[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            x[~mask]
        )
    x = x - np.mean(x)
    x = lowpass_filter(x, fs, cutoff=8)
    #Savitzky-Golay: 11 = velikost okna (11 vzorcev), 2 = stopnja polinoma - kvadratična krivulja
    x = savgol_filter(x, 11, 2) #metoda za glajenje signala, ki poskuša odstraniti šum, hkrati pa ohraniti obliko signala
    return x

#spektrogram
def to_spectrogram(x, fs):
    f, t, Sxx = signal.spectrogram(
        x,
        fs=fs,
        window='hamming', #zmanjša spectral leakage
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10) #skala: bolj vidni šibki spektralni deli
    #normalizacija (0-255)
    vmin = np.percentile(Sxx_db, 5)
    vmax = np.percentile(Sxx_db, 95)
    Sxx_db = np.clip(Sxx_db, vmin, vmax)
    Sxx_norm = (Sxx_db - vmin) / (vmax - vmin + 1e-10)
    img = (Sxx_norm * 255).astype(np.uint8)
    return img, f, t, Sxx_db

def process_file(filepath, output_dir, fs, cls):
    x = np.fromfile(filepath, dtype=np.int16).astype(float)
    x = preprocess_signal(x, fs)
    img, f, t, Sxx_db = to_spectrogram(x, fs)
    filename = os.path.splitext(os.path.basename(filepath))[0]
    out_name = f"{filename}.png"
    out_path = os.path.join(output_dir, out_name)
    plt.figure(figsize=(8,4))
    plt.imshow(Sxx_db, aspect='auto', origin='lower', cmap='viridis', extent=[t.min(), t.max(), f.min(), f.max()])
    plt.ylim(0,30)
    plt.title(f"{filename} - {cls}")
    plt.xlabel("Čas (s)")
    plt.ylabel("Frekvenca (Hz)")
    plt.colorbar(label="Amplituda (dB)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return x, f, t, Sxx_db

classes = ['mahanje', 'mirovanje']

debug_x = None
debug_f = None
debug_t = None
debug_Sxx = None

for cls in classes:
    input_dir = os.path.join(BASE_DIR, "dataset_bin", cls)
    output_dir = os.path.join(BASE_DIR, "dataset_spektrogram", cls)
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)
    for file in files:
        if file.lower().endswith('.bin'):
            filepath = os.path.join(input_dir, file)
            print(f"Processing: {filepath}")
            debug_x, debug_f, debug_t, debug_Sxx = process_file(
                filepath,
                output_dir,
                FS,
                cls
            )