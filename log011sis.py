import numpy as np
import matplotlib.pyplot as plt
from log011 import preberi_bin

def sestavi_podatke(seznam_paketov):
    """Vrne vzorčevalne frekvence po senzorjih in signale"""
    signali = {1: [], 2: [], 3: [], 4: [], 5: []}
    Tpaketi = {1: [], 2: [], 3: [], 4: [], 5: []}
    Nvz = {1: [], 2: [], 3: [], 4: [], 5: []}

    # novo: zadnji timestamp za vsak senzor
    zadnji_ts = {1: None, 2: None, 3: None, 4: None, 5: None}

    for p in seznam_paketov:
        # izračun časa glede na prejšnji paket istega senzorja
        if zadnji_ts[p.id] is not None:
            T = (p.ts - zadnji_ts[p.id]) / 1000.0
        else:
            T = None
        zadnji_ts[p.id] = p.ts
        # IMU (gyro, acc, magnet)
        if p.id in [1, 2, 3]:
            if len(p.data) % 2 =! 0:
                continue
            data_int = np.frombuffer(p.data, dtype=np.int16)
            if p.id == 1:      # gyro
                data_int = data_int * 8.75e-3
            elif p.id == 2:    # acc
                data_int = data_int * 6.125e-5
            elif p.id == 3:    # magnet
                data_int = data_int * 1.5e-3
            if len(data_int) % 3 != 0:
                continue
            vzorci = data_int.reshape(-1, 3)
            signali[p.id].append(vzorci)
            if T is not None and T > 0:
                Tpaketi[p.id].append(T)
                Nvz[p.id].append(len(vzorci))
        # mikrofon
        elif p.id == 4:
            if len(p.data) % 2 != 0:
                continue
            data_int = np.frombuffer(p.data, dtype=np.int16)
            vzorci = data_int.reshape(-1, 1)
            signali[4].append(vzorci)
            if T is not None and T > 0:
                Tpaketi[4].append(T)
                Nvz[4].append(len(vzorci))
        # TOF
        elif p.id == 5:
            if len(p.data) % 2 != 0:
                continue
            data_int = np.frombuffer(p.data, dtype=np.uint16)
            vzorci = data_int.reshape(-1, 1)
            signali[5].append(vzorci)
            if T is not None and T > 0:
                Tpaketi[5].append(T)
                Nvz[5].append(len(vzorci))
    # izračun Fvz
    Fvz_senzor = {}
    for sid in signali:
        if len(Tpaketi[sid]) > 0 and len(Nvz[sid]) > 0:
            Tavg = np.mean(Tpaketi[sid])
            Navg = np.mean(Nvz[sid])
            Fvz_senzor[sid] = Navg / Tavg
        else:
            Fvz_senzor[sid] = 0
    # združi pakete
    for sid in signali:
        if signali[sid]:
            signali[sid] = np.vstack(signali[sid])
        else:
            dim = 3 if sid in [1, 2, 3] else 1
            signali[sid] = np.empty((0, dim))
    return Fvz_senzor, signali

def prikazi_signal(signal: np.ndarray, Fvz: float, naslov: str = "", startInd: int = None, endInd: int = None):
    """Prikaže signal (cel ali del)"""
    if signal.size == 0:
        print("Signal je prazen.")
        return

    start = 0 if startInd is None else startInd
    end = signal.shape[0] if endInd is None else endInd

    signal = signal[start:end]

    t = np.arange(signal.shape[0]) / Fvz

    plt.figure(figsize=(10, 5))

    # 1D signal
    if signal.shape[1] == 1:
        plt.plot(t, signal[:, 0])

        if "mikrofon" in naslov.lower():
            plt.ylabel("Amplituda")
        else:
            plt.ylabel("Razdalja [mm]")

    # 3D signal
    else:
        plt.plot(t, signal[:, 0], label="X")
        plt.plot(t, signal[:, 1], label="Y")
        plt.plot(t, signal[:, 2], label="Z")
        plt.ylabel("Vrednost")

    plt.xlabel("Čas [s]")

    if naslov:
        plt.title(naslov)

    plt.grid(True)
    if signal.shape[1] > 1:
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. branje podatkov
    paketi = preberi_bin("log011.bin")
    # 2. sestavljanje
    Fvz, signali = sestavi_podatke(paketi)

    imena = {
        1: "Gyroskop",
        2: "Pospeškometer",
        3: "Magnetometer",
        4: "Mikrofon",
        5: "TOF"
    }

    for sid in Fvz:
        print(f"{imena[sid]}: {Fvz[sid]:.2f} Hz")

    prikazi_signal(signali[1], Fvz[1], naslov=f"Gyroskop (Fvz={Fvz[1]:.2f} Hz)")
    prikazi_signal(signali[2], Fvz[2], naslov=f"Pospeškometer (Fvz={Fvz[2]:.2f} Hz)")
    prikazi_signal(signali[3], Fvz[3], naslov=f"Magnetometer (Fvz={Fvz[3]:.2f} Hz)")
    prikazi_signal(signali[4], Fvz[4], naslov=f"Mikrofon (Fvz={Fvz[4]:.2f} Hz)")
    prikazi_signal(signali[5], Fvz[5], naslov=f"TOF (Fvz={Fvz[5]:.2f} Hz)")

    # npr. prvih 2 sekundi
    vzorcev_2s_1 = min(int(2 * Fvz[1]), len(signali[1]))
    vzorcev_2s_2 = min(int(2 * Fvz[2]), len(signali[2]))
    vzorcev_2s_3 = min(int(2 * Fvz[3]), len(signali[3]))
    vzorcev_2s_4 = min(int(2 * Fvz[4]), len(signali[4]))
    vzorcev_2s_5 = min(int(2 * Fvz[5]), len(signali[5]))

    prikazi_signal(signali[1], Fvz[1], naslov="Gyroskop (izsek)", startInd=0, endInd=vzorcev_2s_1)
    prikazi_signal(signali[2], Fvz[2], naslov="Pospeškometer (izsek)", startInd=0, endInd=vzorcev_2s_2)
    prikazi_signal(signali[3], Fvz[3], naslov="Magnetometer (izsek)", startInd=0, endInd=vzorcev_2s_3)
    prikazi_signal(signali[4], Fvz[4], naslov="Mikrofon (izsek)", startInd=0, endInd=vzorcev_2s_4)
    prikazi_signal(signali[5], Fvz[5], naslov="TOF (izsek)", startInd=0, endInd=vzorcev_2s_5)
