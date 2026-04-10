import numpy as np
import matplotlib.pyplot as plt
from log011 import preberi_bin, Paket  #funkcija za branje binarnih podatkov

def sestavi_podatke(seznam_paketov):
    """Vrne vzorčevalno frekvenco in signale ločeno po senzorjih"""
    signali = {1: [], 2: [], 3: [], 4: [], 5: []}
    Tpaketi = []
    Nvz_list = []

    for i in range(len(seznam_paketov) - 1):
        p1 = seznam_paketov[i]
        p2 = seznam_paketov[i + 1]

        # čas med paketoma
        T = p2.ts - p1.ts
        Tpaketi.append(T)

        # imu (xyz)
        if p1.id in [1, 2, 3]:
            data_int = np.frombuffer(p1.data, dtype=np.int16)
            # skaliranje glede na senzor
            if p1.id == 1:  # gyro
                data_int = data_int / 120.0

            elif p1.id == 2:  # acc
                data_int = data_int / 20000.0

            elif p1.id == 3:  # magnet
                data_int = data_int / 500.0

            if len(data_int) % 3 != 0:
                continue

            vzorci = data_int.reshape(-1, 3)
            signali[p1.id].append(vzorci)
            Nvz_list.append(len(vzorci))

        # mikrofon
        elif p1.id == 4:
            data_int = np.frombuffer(p1.data, dtype=np.int16)
            vzorci = data_int.reshape(-1, 1)

            signali[4].append(vzorci)
            Nvz_list.append(len(vzorci))

        # tof
        elif p1.id == 5:
            data_int = np.frombuffer(p1.data, dtype=np.uint16)
            vzorci = data_int.reshape(-1, 1)

            signali[5].append(vzorci)
            Nvz_list.append(len(vzorci))

    # izračun vzorčevalne frekvence
    if len(Tpaketi) > 0:
        Fvz = 1.0 / np.mean(Tpaketi)
    else:
        Fvz = 0

    # združevanje vseh paketov
    for sid in signali:
        if signali[sid]:
            signali[sid] = np.vstack(signali[sid])
        else:
            dim = 3 if sid in [1, 2, 3] else 1
            signali[sid] = np.empty((0, dim))

    return Fvz, signali

def prikazi_signal(signal: np.ndarray, naslov: str = "", startInd: int = None, endInd: int = None):
    start = 0 if startInd is None else startInd
    end = signal.shape[0] if endInd is None else endInd

    signal = signal[start:end]
    t = np.arange(signal.shape[0]) / 10

    plt.figure(figsize=(10, 5))

    # 1D signal (mikrofon / TOF)
    if signal.shape[1] == 1:
        plt.plot(t, signal[:, 0])

        if "mikrofon" in naslov.lower():
            plt.ylabel("Amplituda")
        else:
            plt.ylabel("Razdalja (mm)")

    # 3D signal (IMU)
    else:
        plt.plot(t, signal[:, 0], label="X")
        plt.plot(t, signal[:, 1], label="Y")
        plt.plot(t, signal[:, 2], label="Z")
        plt.ylabel("Vrednost")

    plt.xlabel("Vzorec")

    if naslov:
        plt.title(naslov)

    plt.grid(True)
    if signal.shape[1] > 1:
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. branje + dekodiranje (SPO)
    paketi = preberi_bin("log011.bin")

    # 2. sestavljanje podatkov ločeno po senzorjih (SIS)
    Fvz, signali = sestavi_podatke(paketi)

    print(f"Vzorčevalna frekvenca: {Fvz:.2f} Hz")

    # 3. grafi za posamezne senzorje (celotni signal)
    prikazi_signal(signali[1], naslov=f"Gyroskop (Fvz={Fvz:.2f} Hz)")
    prikazi_signal(signali[2], naslov=f"Pospeškometer (Fvz={Fvz:.2f} Hz)")
    prikazi_signal(signali[3], naslov=f"Magnetometer (Fvz={Fvz:.2f} Hz)")
    prikazi_signal(signali[4], naslov=f"Mikrofon (Fvz={Fvz:.2f} Hz)")
    prikazi_signal(signali[5], naslov=f"TOF (Fvz={Fvz:.2f} Hz)")