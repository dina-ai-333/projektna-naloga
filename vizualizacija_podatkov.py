# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:07:18 2026

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt

class Paket:
    def __init__(self, id, ts, data):
        self.id = id
        self.ts = ts
        self.data = data

BYTES_PER_VALUE = {
    1: 2, #žiro
    2: 2, #pospeško
    3: 2 #magneto
}

def sestavi_podatke(seznam_paketov):
    vsi_vzorci = []
    Tpaketi = []
    Nvz_list = []
    for i in range(len(seznam_paketov) - 1):
        p1 = seznam_paketov[i]
        p2 = seznam_paketov[i + 1]
        #čas med paketoma
        T = p2.ts - p1.ts
        Tpaketi.append(T)
        bytes_per_v = BYTES_PER_VALUE[p1.id]
        #št vrednosti v paketu
        st_vrednosti = len(p1.data) // bytes_per_v
        #št vzorcev
        st_vzorcev = st_vrednosti // 3
        Nvz_list.append(st_vzorcev)
        #pretvorba bajtov v int16 (ne vem, če je ravno v 16)
        data_int = np.frombuffer(p1.data, dtype=np.int16)
        vzorci = data_int.reshape(-1, 3)
        vsi_vzorci.append(vzorci)
    #povprečja
    Tavg = np.mean(Tpaketi)
    Navg = np.mean(Nvz_list)
    #vzorčevalna frekv
    Fvz = Navg / Tavg
    #vse pakete združi
    signal = np.vstack(vsi_vzorci)
    return Fvz, signal

def prikazi_signal(signal: np.ndarray, naslov: str = "", startInd: int = None, endInd: int = None):
    start = 0 if startInd is None else startInd
    end = signal.shape[0] if endInd is None else endInd
    signal = signal[start:end]
    plt.figure(figsize=(10, 5))
    if signal.ndim == 1 or signal.shape[1] == 1:
        plt.plot(signal, label="Signal")
    else:
        plt.plot(signal[:, 0], label="X")
        plt.plot(signal[:, 1], label="Y")
        plt.plot(signal[:, 2], label="Z")
    plt.xlabel("Vzorec")
    plt.ylabel("Vrednost")
    if naslov:
        plt.title(naslov)
    plt.grid(True)
    plt.tight_layout()
    plt.show()