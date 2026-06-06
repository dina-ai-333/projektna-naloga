#!/usr/bin/env python3
"""Izpiše vse serijske porte in njihove HWID-je za debug."""
import serial.tools.list_ports

print("=== Vsi serijski porti ===\n")
ports = list(serial.tools.list_ports.comports())
if not ports:
    print("Ni najdenih serijskih portov!")
else:
    for p in ports:
        print(f"Device:      {p.device}")
        print(f"Description: {p.description}")
        print(f"HWID:        {p.hwid}")
        print(f"VID:         {p.vid}")
        print(f"PID:         {p.pid}")
        print()

        # Preveri ujemanje
        hwid_upper = p.hwid.upper() if p.hwid else ""
        vid_hex_match = "VID:0483" in hwid_upper
        pid_hex_match = "PID:5740" in hwid_upper
        vid_int_match = p.vid == 0x0483
        pid_int_match = p.pid == 0x5740
        print(f"  VID hex match (VID:0483): {vid_hex_match}")
        print(f"  PID hex match (PID:5740): {pid_hex_match}")
        print(f"  VID int match (0x0483):   {vid_int_match}")
        print(f"  PID int match (0x5740):   {pid_int_match}")
        print("-" * 40)