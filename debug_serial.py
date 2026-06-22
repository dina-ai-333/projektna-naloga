#!/usr/bin/env python3
"""
Direktna komunikacija s STM32 za debug.
Pošlje ukaz in izpiše surovi odgovor.
"""
import serial
import time

PORT = "COM3"
BAUD = 115200

def send_and_read(ser, cmd, wait=2.0):
    print(f"\n>>> Pošiljam: {cmd!r}")
    ser.reset_input_buffer()
    ser.write((cmd + "\r\n").encode())
    ser.flush()
    
    time.sleep(wait)
    
    raw = ser.read_all()
    print(f"<<< Surovi odgovor ({len(raw)} bytov):")
    print(repr(raw))
    print("--- Kot tekst ---")
    print(raw.decode(errors="replace"))
    print("-" * 40)

try:
    ser = serial.Serial(PORT, BAUD, timeout=3)
    time.sleep(0.5)
    ser.reset_input_buffer()
    print(f"Povezan na {PORT}\n")

    send_and_read(ser, "LIST", wait=2.0)
    
    input("\nPritisni Enter za pošiljanje GET LOG001.BIN (ali Ctrl+C za izhod)...")
    send_and_read(ser, "GET LOG001.BIN", wait=5.0)

    ser.close()
except Exception as e:
    print(f"NAPAKA: {e}")