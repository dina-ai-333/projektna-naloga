import matplotlib.pyplot as plt
import numpy as np

def prikazi_signal(signal, naslov="", startInd=None, endInd=None):

    # Lahko imamo samo del signala
    if startInd is not None and endInd is not None:
        signal = signal[startInd:endInd]

    plt.figure()

    # signal je Nx1, zeto vzamemo prvi stolpec
    plt.plot(signal[:, 0])

    plt.title(naslov)
    plt.xlabel("Vzorec")
    plt.ylabel("Razdalja (mm)")

    plt.grid()

    plt.show()

def sestavi_podatke(seznam_paketov):
    vsi_vzorci = []
    casi = []

    for paket in seznam_paketov:

        # zato ker imamo samo TOF
        if paket.id != 5:
            continue

        # seznam razdalj
        for vrednost in paket.data:
            vsi_vzorci.append([vrednost])   # dobimo Nx1 matriko
            casi.append(paket.ts)

    signal = np.array(vsi_vzorci)

    # izračun vzorčevalne frekvence
    if len(casi) > 1:
        T = np.diff(casi)
        Tavg = np.mean(T)
        Fvz = 1.0 / Tavg
    else:
        Fvz = 0

    return Fvz, signal

if __name__ == "__main__":
    class Paket:
        def __init__(self, id, ts, data):
            self.id = id
            self.ts = ts
            self.data = data

    paketi = [
        Paket(5, 0.0, [100, 101]),
        Paket(5, 0.01, [102, 103]),
    ]

    Fvz, signal = sestavi_podatke(paketi)

    print("Fvz:", Fvz)
    print(signal)

    # GRAF 1 (cel signal)
    prikazi_signal(signal, f"TOF signal (Fvz={Fvz:.2f} Hz)")

    # GRAF 2 (del signala)
    prikazi_signal(signal, "TOF segment", 1, 3)
    
    