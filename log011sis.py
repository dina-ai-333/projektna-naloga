import numpy as np
import matplotlib.pyplot as plt
from log011 import preberi_bin, Paket  #funkcija za branje binarnih podatkov

# število bajtov na vrednost po ID-ju
BYTES_PER_VALUE = {
    1: 2,  # gyro
    2: 2,  # acc
    3: 2   # magnet
}

def sestavi_podatke_ločeno(seznam_paketov):
    """Vrne vzorčevalno frekvenco in signale ločeno po senzorjih"""
    vsi_signali = {1: [], 2: [], 3: []}  # 1=gyro, 2=acc, 3=mag
    Tpaketi = []
    Nvz_list = []

    for i in range(len(seznam_paketov) - 1):
        p1 = seznam_paketov[i]
        p2 = seznam_paketov[i + 1]

        # čas med paketoma
        T = p2.ts - p1.ts
        Tpaketi.append(T)

        bytes_per_v = BYTES_PER_VALUE.get(p1.id, 2)
        st_vrednosti = len(p1.data) // bytes_per_v
        st_vzorcev = st_vrednosti // 3
        Nvz_list.append(st_vzorcev)

        # pretvorba v int16 in reshape
        data_int = np.frombuffer(p1.data, dtype=np.int16)
        vzorci = data_int.reshape(-1, 3)

        if p1.id in vsi_signali:
            vsi_signali[p1.id].append(vzorci)

    # povprečna frekvenca
    Tavg = np.mean(Tpaketi)
    Navg = np.mean(Nvz_list)
    Fvz = Navg / Tavg

    # združijo se paketi po senzorjih
    for sensor_id in vsi_signali:
        if vsi_signali[sensor_id]:
            vsi_signali[sensor_id] = np.vstack(vsi_signali[sensor_id])
        else:
            vsi_signali[sensor_id] = np.empty((0, 3))

    return Fvz, vsi_signali

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
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. branje + dekodiranje (SPO)
    paketi = preberi_bin("log011.bin")

    # 2. sestavljanje podatkov ločeno po senzorjih (SIS)
    Fvz, signali = sestavi_podatke_ločeno(paketi)

    print(f"Vzorčevalna frekvenca: {Fvz:.2f} Hz")

    # 3. grafi za posamezne senzorje (celotni signal)
    prikazi_signal(signali[1], naslov=f"Gyroskop (Fvz={Fvz:.2f} Hz)")
    prikazi_signal(signali[2], naslov=f"Pospeškometer (Fvz={Fvz:.2f} Hz)")
    prikazi_signal(signali[3], naslov=f"Magnetometer (Fvz={Fvz:.2f} Hz)")