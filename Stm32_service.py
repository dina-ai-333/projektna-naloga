#!/usr/bin/env python3
"""
STM32 TCP/IP Service Daemon
Manages STM32 serial connection and exposes control via TCP socket.

STM32 DataLogger serial protocol:
  LIST            → multi-line list of filenames, ends with empty line
  GET <filename>  → raw binary file data, ends with "END\r\n" marker
  DELETE          → deletes all LOG*.BIN files
"""

import socket
import serial
import serial.tools.list_ports
import threading
import logging
import time
import os
import sys
import signal
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
TCP_HOST      = "0.0.0.0"
TCP_PORT      = 5000
STM32_VID     = 0x0483
STM32_PID     = 0x5740
SERIAL_BAUD   = 115200
POLL_INTERVAL = 2            # seconds between USB port-scan cycles
SERIAL_TIMEOUT = 10          # seconds to wait for a serial response
WORK_DIR      = Path(__file__).parent.resolve()

#Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("stm32_service")


#STM32 Connection Manager
class STM32Manager:
    """Thread-safe manager for the STM32 serial connection."""

    def __init__(self):
        self._lock = threading.Lock()
        self._serial: serial.Serial | None = None
        self._port: str | None = None
        self._clients: list = []
        self._clients_lock = threading.Lock()

    # Client registry

    def register_client(self, client):
        with self._clients_lock:
            self._clients.append(client)

    def unregister_client(self, client):
        with self._clients_lock:
            try:
                self._clients.remove(client)
            except ValueError:
                pass

    def broadcast(self, message: str):
        """Push a line to every connected TCP client."""
        with self._clients_lock:
            dead = []
            for c in self._clients:
                try:
                    c.send(message)
                except Exception:
                    dead.append(c)
            for c in dead:
                try:
                    self._clients.remove(c)
                except ValueError:
                    pass

    # Connection state 

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._serial is not None and self._serial.is_open

    @property
    def port(self) -> str | None:
        with self._lock:
            return self._port

    #Low-level serial helpers 

    def connect(self, port: str) -> bool:
        with self._lock:
            try:
                ser = serial.Serial(port, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
                # Flush any garbage from the STM32 boot message
                time.sleep(0.3)
                ser.reset_input_buffer()
                self._serial = ser
                self._port = port
                log.info("Connected to STM32 on %s", port)
                return True
            except Exception as e:
                log.error("Failed to open %s: %s", port, e)
                self._serial = None
                self._port = None
                return False

    def disconnect(self):
        with self._lock:
            if self._serial:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
                self._port = None

    def _write(self, cmd: str):
        """Send a command with CRLF line ending (matches STM32 CRLF expectation)."""
        self._serial.write((cmd + "\r\n").encode())
        self._serial.flush()

    def _readline(self) -> str:
        """Read one line, strip CR/LF, decode."""
        raw = self._serial.readline()
        return raw.decode(errors="replace").strip()

    #  Protocol implementations 

    def list_files(self) -> list[str]:
        """
        Send LIST, parse response until '>' prompt.
        STM32 response format:
            Listing files...\r\n
            \r\n
            Volume is FAT32\r\n
            LOG001.BIN     2193\r\n
            ...
            >\r\n          <- STM32 prompt = end of response
        """
        with self._lock:
            if not self._serial or not self._serial.is_open:
                raise IOError("STM32 not connected")
            try:
                self._write("LIST")
                files = []
                while True:
                    line = self._readline()
                    if line == ">":
                        break
                    # Skip header/info lines
                    if not line or line.startswith("Listing") or line.startswith("Volume"):
                        continue
                    # Each file line: "LOG001.BIN     2193"
                    parts = line.split()
                    if parts and parts[0].upper().endswith(".BIN"):
                        files.append(parts[0])
                log.info("LIST returned %d file(s)", len(files))
                return files
            except serial.SerialException as e:
                self._serial = None
                self._port = None
                raise IOError("STM32 disconnected during LIST") from e

    def get_file(self, filename: str) -> bytes:
        """
        Send GET <filename>, receive binary data.
        STM32 response protocol:
            Transferring: LOG116.BIN\r\n
            File size: 192 bytes\r\n
            <exactly N bytes of binary data>
            Transfer complete: 192 bytes\r\n
        """
        with self._lock:
            if not self._serial or not self._serial.is_open:
                raise IOError("STM32 not connected")
            try:
                self._write(f"GET {filename}")

                # Line 1: "Transferring: <filename>"
                line1 = self._readline()
                if not line1.startswith("Transferring"):
                    raise IOError(f"Unexpected GET response: {line1!r}")

                # Line 2: "File size: 192 bytes"
                line2 = self._readline()
                if not line2.startswith("File size:"):
                    raise IOError(f"Unexpected size line: {line2!r}")
                # Parse size — "File size: 192 bytes" -> 192
                size = int(line2.split(":")[1].strip().split()[0])
                log.info("GET %s: expecting %d bytes", filename, size)

                # Read exactly N bytes of binary data
                data = b""
                remaining = size
                while remaining > 0:
                    chunk = self._serial.read(min(remaining, 4096))
                    if not chunk:
                        raise IOError(f"Timeout reading file data for {filename} "
                                      f"(got {len(data)}/{size} bytes)")
                    data += chunk
                    remaining -= len(chunk)

                # Drain trailing lines: "Transfer complete: N bytes" and ">" prompt
                while True:
                    line = self._readline()
                    log.debug("GET trailer: %r", line)
                    if line == ">" or line == "":
                        break

                log.info("Received %s: %d bytes", filename, len(data))
                return data
            except serial.SerialException as e:
                self._serial = None
                self._port = None
                raise IOError("STM32 disconnected during GET") from e

    def delete_all(self):
        """Send DELETE, drain response until OK confirmation or > prompt."""
        with self._lock:
            if not self._serial or not self._serial.is_open:
                raise IOError("STM32 not connected")
            try:
                self._write("DELETE")
                for _ in range(20):  # max 20 lines to avoid infinite loop
                    line = self._readline()
                    log.info("DELETE response: %s", line)
                    if line == ">" or line.startswith("OK"):
                        break
            except serial.SerialException as e:
                self._serial = None
                self._port = None
                raise IOError("STM32 disconnected during DELETE") from e


#Hot-plug Monitor 
class HotplugMonitor(threading.Thread):
    """Polls USB serial ports every POLL_INTERVAL seconds."""

    def __init__(self, manager: STM32Manager):
        super().__init__(daemon=True, name="HotplugMonitor")
        self.manager = manager
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def _find_stm32_port(self) -> str | None:
        try:
            for port in serial.tools.list_ports.comports():
                # Use integer comparison — works on both Windows and Linux
                # (Windows HWID format: "USB VID:PID=0483:5740", Linux: "USB VID:0483 PID:5740")
                if port.vid == STM32_VID and port.pid == STM32_PID:
                    return port.device
        except Exception as e:
            log.debug("Port scan error: %s", e)
        return None

    def run(self):
        log.info("HotplugMonitor started (VID=%04X PID=%04X)", STM32_VID, STM32_PID)
        while not self._stop_evt.is_set():
            stm32_port = self._find_stm32_port()

            if stm32_port and not self.manager.is_connected:
                log.info("STM32 detected on %s — connecting…", stm32_port)
                if self.manager.connect(stm32_port):
                    self.manager.broadcast(f"STM32 detected at {stm32_port}")

            elif not stm32_port and self.manager.is_connected:
                log.info("STM32 unplugged")
                self.manager.disconnect()
                self.manager.broadcast("STM32 has disconnected")

            self._stop_evt.wait(POLL_INTERVAL)


# Per-client TCP handler 
class ClientHandler(threading.Thread):
    """One thread per TCP connection."""

    def __init__(self, conn: socket.socket, addr, manager: STM32Manager):
        super().__init__(daemon=True, name=f"Client-{addr}")
        self.conn = conn
        self.addr = addr
        self.manager = manager
        self._send_lock = threading.Lock()

    def send(self, msg: str):
        with self._send_lock:
            try:
                self.conn.sendall((msg + "\n").encode())
            except Exception:
                pass

    def run(self):
        self.manager.register_client(self)
        log.info("Client connected: %s", self.addr)

        # Greeting
        if self.manager.is_connected:
            self.send(f"Connected to SPO STM32 service - STM32 at {self.manager.port}")
        else:
            self.send("Connected to SPO STM32 service - No STM32 detected")

        try:
            buf = b""
            while True:
                chunk = self.conn.recv(1024)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    cmd = line.decode(errors="replace").strip()
                    if cmd:
                        response = self._handle(cmd)
                        if response:
                            self.send(response)
        except Exception as e:
            log.debug("Client %s read error: %s", self.addr, e)
        finally:
            self.manager.unregister_client(self)
            try:
                self.conn.close()
            except Exception:
                pass
            log.info("Client disconnected: %s", self.addr)

    # Command dispatcher 

    def _handle(self, cmd: str) -> str:
        log.info("[%s] CMD: %s", self.addr, cmd)

        # STATUS — works even without STM32
        if cmd == "STATUS":
            if self.manager.is_connected:
                return "STM32 is connected"
            return "STM32 is not connected"

        # All other commands require a connected STM32
        if not self.manager.is_connected:
            return "FAIL: STM32 is not connected"

        try:
            # GET_LAST 
            if cmd == "GET_LAST":
                files = self.manager.list_files()
                if not files:
                    return "FAIL: No files on STM32"
                self._fetch_and_save(files[-1])
                return "Last file from STM32 has been processed"

            # GET_ALL 
            elif cmd == "GET_ALL":
                files = self.manager.list_files()
                if not files:
                    return "FAIL: No files on STM32"
                for f in files:
                    self._fetch_and_save(f)
                return "All files from STM32 are processed"

            # GET_FILE|name 
            elif cmd.startswith("GET_FILE|"):
                filename = cmd[len("GET_FILE|"):].strip()
                if not filename:
                    return "FAIL: No filename specified"
                return self._fetch_and_save(filename)

            # DELETE 
            elif cmd == "DELETE":
                self.manager.delete_all()
                return "All files on STM32 are deleted"

            else:
                return f"FAIL: Unknown command '{cmd}'"

        except IOError as e:
            log.warning("IO error handling '%s': %s", cmd, e)
            return f"FAIL: {e}"
        except Exception as e:
            log.error("Unexpected error handling '%s': %s", cmd, e, exc_info=True)
            return f"FAIL: Internal error — {e}"

    def _fetch_and_save(self, filename: str) -> str:
        data = self.manager.get_file(filename)
        dest = WORK_DIR / filename
        dest.write_bytes(data)
        log.info("Saved %s (%d bytes) → %s", filename, len(data), dest)
        return f"File {filename} from STM32 has been processed"


# TCP Server 
class TCPServer(threading.Thread):

    def __init__(self, manager: STM32Manager):
        super().__init__(daemon=True, name="TCPServer")
        self.manager = manager
        self._stop_evt = threading.Event()
        self._sock: socket.socket | None = None

    def stop(self):
        self._stop_evt.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def run(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((TCP_HOST, TCP_PORT))
        self._sock.listen(10)
        self._sock.settimeout(1.0)
        log.info("TCP server listening on %s:%d", TCP_HOST, TCP_PORT)

        while not self._stop_evt.is_set():
            try:
                conn, addr = self._sock.accept()
                ClientHandler(conn, addr, self.manager).start()
            except socket.timeout:
                continue
            except OSError:
                break

        log.info("TCP server stopped")


# Entry point 
def main():
    log.info("STM32 service starting (PID %d)", os.getpid())
    log.info("Work directory: %s", WORK_DIR)

    manager = STM32Manager()
    monitor = HotplugMonitor(manager)
    server  = TCPServer(manager)

    monitor.start()
    server.start()

    stop_event = threading.Event()

    def _shutdown(signum, frame):
        log.info("Signal %d received — shutting down…", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    stop_event.wait()

    server.stop()
    monitor.stop()
    manager.disconnect()
    log.info("STM32 service stopped")


if __name__ == "__main__":
    main()