import struct

class Paket:
    def __init__(self, id, ts, data):
        self.id = id
        self.ts = ts
        self.data = data

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

def preberi_bin(file_path) -> list:
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

                    if chunk_id in [1, 2, 3]:
                        paketi.append(Paket(chunk_id, timestamp, chunk_data))

                    pos += 4 + chunk_size

                buffer = buffer[sync_index + 2:]

    return paketi