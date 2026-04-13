from Get_bin import preberi_bin
import matplotlib.pyplot as plt
import numpy as np

GYRO = 1
ACC = 2
MAG = 3
MIC = 4
TOF = 5

def sestavi_podatke(seznam_paketov):
    signali = {1: [], 2: [], 3: [], 4: [], 5: []}
    Tpaketi = {1: [], 2: [], 3: [], 4: [], 5: []}
    Nvz = {1: [], 2: [], 3: [], 4: [], 5: []}

    # timestamp za vsak senzor
    time_st = {1: None, 2: None, 3: None, 4: None, 5: None}

    for p in seznam_paketov:
        # izračun časa na podalgi prejšnjega paketa iz istega senzorja
        if time_st[p.id] is not None:
            T = (p.ts - time_st[p.id])
            
        else:
            T = None
        time_st[p.id] = p.ts
        # IMU
        if p.id in [GYRO, ACC, MAG]:
            if len(p.data) % 2 != 0:
                continue
            
            data_int = np.frombuffer(p.data, dtype=np.int16)
            
            if p.id == GYRO:
                data_int = data_int * 8.75e-3
                
            elif p.id == ACC:    
                data_int = data_int * 6.125e-5
                
            elif p.id == MAG:
                data_int = data_int * 1.5e-3
                
            if len(data_int) % 3 != 0:
                continue
            
            vzorci = data_int.reshape(-1, 3)
            signali[p.id].append(vzorci)
            
            if T is not None and T > 0:
                Tpaketi[p.id].append(T)
                Nvz[p.id].append(len(vzorci))
        # mic
        elif p.id == MIC:
            if len(p.data) % 2 != 0:
                continue
            
            data_int = np.frombuffer(p.data, dtype=np.int16)
            vzorci = data_int.reshape(-1, 1)
            signali[p.id].append(vzorci)
            
            if T is not None and T > 0:
                Tpaketi[p.id].append(T)
                Nvz[p.id].append(len(vzorci))
        # TOF
        elif p.id == TOF:
            if len(p.data) % 2 != 0:
                continue
            
            data_int = np.frombuffer(p.data, dtype=np.uint16)
            vzorci = data_int.reshape(-1, 1)
            signali[p.id].append(vzorci)
            if T is not None and T > 0:
                Tpaketi[p.id].append(T)
                Nvz[p.id].append(len(vzorci))
    # Fvz
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
            plt.ylabel("Amplitude")
        else:
            plt.ylabel("Distance [mm]")

    # 3D signal
    else:
        plt.plot(t, signal[:, 0], label="X")
        plt.plot(t, signal[:, 1], label="Y")
        plt.plot(t, signal[:, 2], label="Z")
        plt.ylabel("Value")

    plt.xlabel("Time [s]")

    if naslov:
        plt.title(naslov)

    plt.grid(True)
    if signal.shape[1] > 1:
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. branje podatkov
    paketi = preberi_bin("log004.bin")
    # 2. sestavljanje
    Fvz, signal = sestavi_podatke(paketi)

    moduli = {
        1: "Gyroscope",
        2: "Accelerometer",
        3: "Magnetometer",
        4: "Microphone",
        5: "TOF"
    }

    for sid in Fvz:
        print(f"{moduli[sid]}: {Fvz[sid]:.2f} Hz")

    prikazi_signal(signal[1], Fvz[1], naslov=f"Gyro\n (Fvz={Fvz[1]:.2f} Hz)")
    prikazi_signal(signal[2], Fvz[2], naslov=f"Accel\n (Fvz={Fvz[2]:.2f} Hz)")
    prikazi_signal(signal[3], Fvz[3], naslov=f"Magnet\n (Fvz={Fvz[3]:.2f} Hz)")
    prikazi_signal(signal[4], Fvz[4], naslov=f"Mic\n (Fvz={Fvz[4]:.2f} Hz)")
    prikazi_signal(signal[5], Fvz[5], naslov=f"TOF\n (Fvz={Fvz[5]:.2f} Hz)")

    # izsek za par sekund
    segment_gy = min(int(4 * Fvz[1]), len(signal[1]))
    segment_ac = min(int(4 * Fvz[2]), len(signal[2]))
    segment_mg = min(int(4 * Fvz[3]), len(signal[3]))
    segment_mc = min(int(4 * Fvz[4]), len(signal[4]))
    segment_tof = min(int(4 * Fvz[5]), len(signal[5]))

    prikazi_signal(signal[1], Fvz[1], naslov="Gyroscope (segment)", startInd=0, endInd=segment_gy)
    prikazi_signal(signal[2], Fvz[2], naslov="Accelerometer (segment)", startInd=0, endInd=segment_ac)
    prikazi_signal(signal[3], Fvz[3], naslov="Magnetometer (segment)", startInd=0, endInd=segment_mg)
    prikazi_signal(signal[4], Fvz[4], naslov="Microphone (segment)", startInd=0, endInd=segment_mc)
    prikazi_signal(signal[5], Fvz[5], naslov="TOF (segment)", startInd=0, endInd=segment_tof)