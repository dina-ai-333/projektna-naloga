import numpy as np
from scipy.signal import butter, filtfilt


def remove_zeros(x):
    x = x.copy()

    mask = (x == 0)

    if np.any(mask):
        x[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            x[~mask]
        )

    return x


def lowpass_filter(x, fs, cutoff=5, order=4):

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low')

    return filtfilt(b, a, x)


def normalize(x):

    std = np.std(x)

    if std < 1e-8:
        return x - np.mean(x)

    return (x - np.mean(x)) / std


def preprocess_signal(x, fs):

    x = remove_zeros(x)

    x = x - np.mean(x)

    x = lowpass_filter(x, fs, cutoff=5)

    x = normalize(x)

    return x