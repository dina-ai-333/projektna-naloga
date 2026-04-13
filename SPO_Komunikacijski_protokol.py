import serial
import struct

#razred za pakete
class Paket:
    def __init__(self, id, ts, data):
        self.id = id
        self.ts = ts
        self.data = data

ser = serial.Serial('COM3', 115200, timeout=0.1)
ser.reset_input_buffer()

print("Listening...")

def unstuff_bytes(data):
    result = bytearray()
    i = 0

    while i < len(data):
        if data[i] == 0xFE:
            if i + 1 >= len(data):
                break
            result.append(data[i] ^ data[i+1])
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


        
# za sis rabimo branje iz .BIN
def decode_bin_file(filename):
    with open(filename, "rb") as f:
        data_stream = f.read()

    buffer = bytearray(data_stream)
    paketi = []

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
            continue

        if packet_size > 2000:
            buffer = buffer[sync_index + 2:]
            continue

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
            chunk_size = struct.unpack('<H', payload[pos+1:pos+3])[0] + 1
            chunk_data = payload[pos+4:pos+4+chunk_size]

            if chunk_id == 0x05 and chunk_size >= 2:
                distance = struct.unpack('<H', chunk_data[0:2])[0]
                paketi.append(Paket(5, timestamp / 1000.0, [distance]))

            pos += 4 + chunk_size

        buffer = buffer[sync_index + 2:]

    return paketi


if __name__ == "__main__":

    buffer = bytearray()
    paketi = []

    while True:
       data = ser.read(256)

       if data:
           buffer.extend(data)

      
       if len(buffer) > 5000:
           print("Buffer overflow -> reset")
           buffer.clear()
           continue

       while True:
           sync_index = buffer.find(b'\xFF\xFF')

           if sync_index == -1:
               break

           if len(buffer) < sync_index + 4:
               break

           packet_counter = buffer[sync_index + 2]

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
               print("Invalid packet size -> skip")
               buffer = buffer[sync_index + 2:]
               break

           total_needed = 6 + packet_size + 2

           if len(unstuffed) < total_needed:
               break  # čakamo še podatke

           payload = unstuffed[:total_needed]

           received_crc = struct.unpack('<H', payload[-2:])[0]
           computed_crc = crc16(payload[:-2])

           if received_crc != computed_crc:
               print("CRC ERROR")
               buffer = buffer[sync_index + 2:]
               continue

           print("\n======================================")
           print(f"Packet #{packet_counter}")
           print(f"Timestamp: {timestamp} ms")
           print(f"Packet size: {packet_size}")
           print("--------------------------------------")

           pos = 6

           while pos < len(payload) - 2:

               chunk_id = payload[pos]
               chunk_size = struct.unpack('<H', payload[pos+1:pos+3])[0] + 1
               reserved = payload[pos+3]
               chunk_data = payload[pos+4:pos+4+chunk_size]

               if chunk_id == 0x01:
                   print("GYROSCOPE (ID=1):")
                   for i in range(0, chunk_size, 6):
                       if i + 6 > len(chunk_data):
                           break
                       x, y, z = struct.unpack('<hhh', chunk_data[i:i+6])
                       print(f"  x={x:6} y={y:6} z={z:6}")

               elif chunk_id == 0x02:
                   print("\nACCELEROMETER (ID=2):")
                   for i in range(0, chunk_size, 6):
                       if i + 6 > len(chunk_data):
                           break
                       x, y, z = struct.unpack('<hhh', chunk_data[i:i+6])
                       print(f"  x={x:6} y={y:6} z={z:6}")

               elif chunk_id == 0x03:
                   print("\nMAGNETOMETER (ID=3):")
                   for i in range(0, chunk_size, 6):
                       if i + 6 > len(chunk_data):
                           break
                       x, y, z = struct.unpack('<hhh', chunk_data[i:i+6])
                       print(f"  x={x:6} y={y:6} z={z:6}")

               elif chunk_id == 0x04:
                   print("\nMICROPHONE (ID=4):")
                   for i in range(0, chunk_size, 2):
                       if i + 2 > len(chunk_data):
                           break
                       mic = struct.unpack('<h', chunk_data[i:i+2])[0]
                       print(f"  mic = {mic}")

               elif chunk_id == 0x05:
                   print("\nTOF SENSOR (ID=5):")
                   if chunk_size >= 2:
                       distance = struct.unpack('<H', chunk_data[0:2])[0]
                       print(f"  distance = {distance} mm")
                       # dodano za shranjevanje podatkov
                       paketi.append(Paket(5, timestamp / 1000.0, [distance]))
               else:
                   print(f"\nUNKNOWN SENSOR ID={chunk_id}")

               pos += 4 + chunk_size

           print("\n======================================")

           buffer = buffer[sync_index + 2:]