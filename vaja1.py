import serial
import time
import struct
import numpy as np
import matplotlib.pyplot as plt

# SERIAL DOWNLOAD (STM → PC)
def prenesi_bin(port="COM3", filename="LOG002.BIN", filesize=190810):
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)

    ser.reset_input_buffer()

    cmd = f"GET {filename}\n"
    ser.write(cmd.encode())

    print("Downloading...")

    received = 0

    with open(filename, "wb") as f:
        while received < filesize:
            data = ser.read(min(1024, filesize - received))

            if data:
                f.write(data)
                received += len(data)
                print(f"{received}/{filesize} bytes")
            else:
                print("Timeout - konec prenosa")
                break

    ser.close()
    print("Download končan\n")


# PODATKOVNE STRUKTURE
class Paket:
    def __init__(self, id, ts, data):
        self.id = id
        self.ts = ts
        self.data = data


# BYTE UNSTUFFING
def unstuff_bytes(data):
    result = bytearray()
    i = 0
    while i < len(data):
        if data[i] == 0xFE:
            if i + 1 >= len(data):
                break
            result.append(data[i] ^ data[i + 1])
            i += 2
        else:
            result.append(data[i])
            i += 1
    return bytes(result)


# CRC16
def crc16(data):
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


# BIN PARSER
def preberi_bin(file_path):
    paketi = []
    buffer = bytearray()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(256)
            if not data:
                break

            buffer.extend(data)

            while True:
                sync_index = buffer.find(b'\xFF\xFF')
                if sync_index == -1:
                    break

                if len(buffer) < sync_index + 4:
                    break

                raw_payload = buffer[sync_index + 3:]
                unstuffed = unstuff_bytes(raw_payload)

                if len(unstuffed) < 6:
                    break

                try:
                    timestamp = struct.unpack('<I', unstuffed[0:4])[0]
                    packet_size = struct.unpack('<H', unstuffed[4:6])[0] + 1
                except:
                    buffer = buffer[sync_index + 2:]
                    break

                if packet_size > 2000:
                    buffer = buffer[sync_index + 2:]
                    break

                total_needed = 6 + packet_size + 2
                if len(unstuffed) < total_needed:
                    break

                payload = unstuffed[:total_needed]

                received_crc = struct.unpack('<H', payload[-2:])[0]
                computed_crc = crc16(payload[:-2])

                if received_crc != computed_crc:
                    buffer = buffer[sync_index + 2:]
                    continue

                pos = 6
                while pos < len(payload) - 2:
                    chunk_id = payload[pos]
                    chunk_size = struct.unpack('<H', payload[pos + 1:pos + 3])[0] + 1
                    chunk_data = payload[pos + 4:pos + 4 + chunk_size]

                    if chunk_id in [1, 2, 3, 4, 5]:
                        paketi.append(Paket(chunk_id, timestamp, chunk_data))

                    pos += 4 + chunk_size

                buffer = buffer[sync_index + 2:]

    return paketi


# SESTAVI PODATKE
def sestavi_podatke(seznam_paketov):
    signali = {1: [], 2: [], 3: [], 4: [], 5: []}
    Tpaketi = []

    for i in range(len(seznam_paketov) - 1):
        p1 = seznam_paketov[i]
        p2 = seznam_paketov[i + 1]

        Tpaketi.append(p2.ts - p1.ts)

        if p1.id in [1, 2, 3]:
            data_int = np.frombuffer(p1.data, dtype=np.int16)
            if len(data_int) % 3 != 0:
                continue
            signali[p1.id].append(data_int.reshape(-1, 3))

        elif p1.id == 4:
            data_int = np.frombuffer(p1.data, dtype=np.int16)
            signali[4].append(data_int.reshape(-1, 1))

        elif p1.id == 5:
            data_int = np.frombuffer(p1.data, dtype=np.uint16)
            signali[5].append(data_int.reshape(-1, 1))

    Fvz = 1.0 / np.mean(Tpaketi) if Tpaketi else 0

    for sid in signali:
        if signali[sid]:
            signali[sid] = np.vstack(signali[sid])
        else:
            dim = 3 if sid in [1, 2, 3] else 1
            signali[sid] = np.empty((0, dim))

    return Fvz, signali


# PLOT
def prikazi_signal(signal, naslov=""):
    plt.figure(figsize=(10, 5))

    if signal.shape[1] == 1:
        plt.plot(signal[:, 0])
    else:
        plt.plot(signal[:, 0], label="X")
        plt.plot(signal[:, 1], label="Y")
        plt.plot(signal[:, 2], label="Z")
        plt.legend()

    plt.title(naslov)
    plt.xlabel("Vzorec")
    plt.grid(True)
    plt.show()


# MAIN
if __name__ == "__main__":

    # pomembno: prilagodi
    PORT = "COM3"
    FILESIZE = 190810  # iz LIST

    # 1. download
    prenesi_bin(PORT, "LOG002.BIN", FILESIZE)

    # 2. decode
    paketi = preberi_bin("LOG002.BIN")
    print(f"Št. paketov: {len(paketi)}")

    # 3. obdelava
    Fvz, signali = sestavi_podatke(paketi)
    print(f"Fvz: {Fvz:.2f} Hz")

    # 4. grafi
    prikazi_signal(signali[1], "Gyro")
    prikazi_signal(signali[2], "Accel")
    prikazi_signal(signali[3], "Mag")
    prikazi_signal(signali[4], "Mic")
    prikazi_signal(signali[5], "TOF")