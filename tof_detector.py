import numpy as np
import serial
import struct
from collections import deque

from Get_bin import Paket

TOF = 5
COM_PORT = "COM3"
BAUDRATE = 115200

# Prag za zaznavanje mahanja - std razdalje mora biti > tega
STD_PRAG = 100  # mm


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


class TOFDetector:

    def __init__(self, model, mean, std, port=COM_PORT, baudrate=BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        # model še vedno shranjujemo ampak ga ne rabimo za mahanje
        self.model = model
        self.mean = mean
        self.std = std

    def wait_for_wave(self):
        ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        ser.reset_input_buffer()

        buffer = bytearray()
        # drseče okno zadnjih ~20 vzorcev (~1 sekunda)
        tof_window = deque(maxlen=20)
        last_valid = None

        print(f"Poslušam na {self.port}...")

        try:
            while True:
                data = ser.read(256)
                if data:
                    buffer.extend(data)

                if len(buffer) > 5000:
                    buffer.clear()
                    continue

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
                        packet_size = struct.unpack('<H', unstuffed[4:6])[0] + 1
                    except Exception:
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

                        if chunk_id == TOF and chunk_size >= 2:
                            distance = struct.unpack('<H', chunk_data[0:2])[0]

                            if distance == 65535:
                                distance = last_valid if last_valid else 0
                            else:
                                last_valid = distance

                            tof_window.append(distance)

                            # ko imamo dovolj vzorcev → preveri std
                            if len(tof_window) == 20:
                                std_val = np.std(tof_window)
                                print(f"  std={std_val:.0f}mm", end='\r')

                                if std_val > STD_PRAG:
                                    print(f"\n  Mahanje zaznano! (std={std_val:.0f}mm)")
                                    ser.close()
                                    return True

                        pos += 4 + chunk_size

                    buffer = buffer[sync_index + 2:]

        except KeyboardInterrupt:
            ser.close()
            raise