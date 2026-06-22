import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

def interpolate_missing(x):
    x = x.copy()
    if np.mean(x == 0) < 0.5:
        x[x == 0] = np.nan

        mask = np.isnan(x)
        if np.any(mask):
            x[mask] = np.interp(
                np.flatnonzero(mask),
                np.flatnonzero(~mask),
                x[~mask]
            )
        x = x - np.mean(x)
        return x

def lowpass_filter(x, fs, cutoff=5, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, x)

def smooth_signal(x, window=11, poly=2):
    return savgol_filter(x, window, poly)

def normalize(x):
    std = np.std(x)
    if std < 1e-8:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def preprocess_signal(x, fs):
    x = x.copy()
    #čiščenje
    x = interpolate_missing(x)
    #centriranje
    x = x - np.mean(x)
    #low pass
    x = lowpass_filter(x, fs, cutoff=5)
    #rahlo glajenje
    #x = smooth_signal(x, window=9, poly=2)
    #normalizacija
    x = normalize(x)
    return x