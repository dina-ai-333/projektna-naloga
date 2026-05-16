import os
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter, butter, filtfilt
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FS = 50

WINDOW_SEC = 2.0
OVERLAP = 0.5

NPERSEG = 64
NOVERLAP = 48
NFFT = 128

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
    #Savitzky manjši window (da ne zbriše strukture)
    x = savgol_filter(x, 11, 2)
    return x

#segmentacija
def split_signal(x, fs, window_sec=2.0, overlap=0.5):
    step = int(window_sec * fs * (1 - overlap))
    size = int(window_sec * fs)
    segments = []
    for start in range(0, len(x) - size + 1, step):
        segments.append(x[start:start + size])
    return segments

#spektrogram
def to_spectrogram(x, fs):
    f, t, Sxx = signal.spectrogram(
        x,
        fs=fs,
        window='hamming',
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    #normalizacija (0-255)
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
        img, f, t, Sxx_db = to_spectrogram(seg, fs)
        filename = os.path.splitext(os.path.basename(filepath))[0]
        out_name = f"{filename}_{i}.png"
        out_path = os.path.join(output_dir, out_name)
        Image.fromarray(img).save(out_path)
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
                FS
            )

#grafi
results_dir = os.path.join(BASE_DIR, "rezultati")
os.makedirs(results_dir, exist_ok=True)

plt.figure(figsize=(10,4))
plt.plot(debug_x)
plt.title("Filtriran ToF signal")
plt.xlabel("Vzorec")
plt.ylabel("Amplituda")
plt.grid(True)

plt.savefig(os.path.join(results_dir, "signal.png"), dpi=300)
plt.close()

plt.figure(figsize=(8,4))
plt.imshow(
    debug_Sxx,
    aspect='auto',
    origin='lower',
    cmap='viridis',
    extent=[debug_t.min(), debug_t.max(), debug_f.min(), debug_f.max()]
)

plt.title("Spektrogram")
plt.xlabel("Čas [s]")
plt.ylabel("Frekvenca [Hz]")
plt.colorbar()

plt.savefig(os.path.join(results_dir, "spektrogram.png"), dpi=300)
plt.close()