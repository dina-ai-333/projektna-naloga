#!/usr/bin/env python3
"""
Preprost TCP odjemalec za testiranje STM32 storitve.
Uporaba: python client.py
"""

import socket
import threading
import sys

HOST = "127.0.0.1"
PORT = 5000

def receive_loop(sock):
    """V ozadju sprejema push sporočila (npr. STM32 detected)."""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                print("\n[Povezava zaprta]")
                break
            print(data.decode(errors="replace"), end="", flush=True)
        except Exception:
            break

def main():
    print(f"Povezujem na {HOST}:{PORT} ...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
    except ConnectionRefusedError:
        print("NAPAKA: Storitev ne teče! Zaženi najprej python Stm32_service.py")
        sys.exit(1)

    # Nit za sprejemanje sporočil od storitve
    t = threading.Thread(target=receive_loop, args=(sock,), daemon=True)
    t.start()

    print("Vpisi ukaz (STATUS, GET_LAST, GET_ALL, GET_FILE|ime, DELETE) ali 'quit':\n")

    try:
        while True:
            cmd = input()
            if cmd.strip().lower() == "quit":
                break
            sock.sendall((cmd.strip() + "\n").encode())
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        sock.close()
        print("\nOdjavljen.")

if __name__ == "__main__":
    main()