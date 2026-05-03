import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from spo_naloga import preberi_bin


def sestavi_podatke(seznam_paketov):
    signali = {1: [], 2: [], 3: [], 4: [], 5: []}
    Tpaketi = {1: [], 2: [], 3: [], 4: [], 5: []}
    Nvz = {1: [], 2: [], 3: [], 4: [], 5: []}
    zadnji_ts = {1: None, 2: None, 3: None, 4: None, 5: None}

    for p in seznam_paketov:
        if zadnji_ts[p.id] is not None:
            T = (p.ts - zadnji_ts[p.id]) / 1000.0
        else:
            T = None
        zadnji_ts[p.id] = p.ts

        if p.id in [1, 2, 3]:
            if len(p.data) % 2 != 0:
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

        elif p.id == 4:  # mikrofon
            if len(p.data) % 2 != 0:
                continue
            data_int = np.frombuffer(p.data, dtype=np.int16)
            vzorci = data_int.reshape(-1, 1)
            signali[4].append(vzorci)

            if T is not None and T > 0:
                Tpaketi[4].append(T)
                Nvz[4].append(len(vzorci))

        elif p.id == 5:  # TOF
            if len(p.data) % 2 != 0:
                continue
            data_int = np.frombuffer(p.data, dtype=np.uint16)

            # popravek napačnih vrednosti
            filtered = []
            last_valid = None
            for v in data_int:
                if v == 65535:
                    filtered.append(last_valid if last_valid is not None else 0)
                else:
                    filtered.append(v)
                    last_valid = v

            data_int = np.array(filtered, dtype=np.uint16)
            vzorci = data_int.reshape(-1, 1)
            signali[5].append(vzorci)

            if T is not None and T > 0:
                Tpaketi[5].append(T)
                Nvz[5].append(len(vzorci))

    Fvz_senzor = {}
    for sid in signali:
        if len(Tpaketi[sid]) > 0:
            Tavg = np.mean(Tpaketi[sid])
            Navg = np.mean(Nvz[sid])
            Fvz_senzor[sid] = Navg / Tavg
        else:
            Fvz_senzor[sid] = 0

    for sid in signali:
        if signali[sid]:
            signali[sid] = np.vstack(signali[sid])
        else:
            dim = 3 if sid in [1, 2, 3] else 1
            signali[sid] = np.empty((0, dim))

    return Fvz_senzor, signali


def shrani_tof_v_csv(signal, Fvz, filename, label):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["čas", "razdalja", "label"])

        for i in range(len(signal)):
            cas = i / Fvz if Fvz > 0 else 0
            razdalja = signal[i][0]
            writer.writerow([cas, razdalja, label])


def prikazi_signal(signal, Fvz, naslov=""):
    if signal.size == 0:
        return

    t = np.arange(signal.shape[0]) / Fvz
    plt.figure(figsize=(8, 4))
    plt.plot(t, signal[:, 0])
    plt.title(naslov)
    plt.xlabel("Čas [s]")
    plt.ylabel("Razdalja [mm]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    mapa_bin = "podatki_bin"
    mapa_csv = "csv_podatki"

    os.makedirs(mapa_csv, exist_ok=True)

    for ime_datoteke in os.listdir(mapa_bin):
        if not ime_datoteke.endswith(".bin"):
            continue

        pot = os.path.join(mapa_bin, ime_datoteke)
        print(f"Obdelujem: {ime_datoteke}")

        paketi = preberi_bin(pot)
        Fvz, signali = sestavi_podatke(paketi)

        # določanje labela iz imena
        if "blizu" in ime_datoteke:
            label = 0
        elif "srednje" in ime_datoteke:
            label = 1
        elif "dalec" in ime_datoteke:
            label = 2
        else:
            print("Neznan label")
            continue

        ime_csv = ime_datoteke.replace(".bin", ".csv")
        pot_csv = os.path.join(mapa_csv, ime_csv)

        shrani_tof_v_csv(signali[5], Fvz[5], pot_csv, label)

    print("Vsi CSV-ji so generirani!")