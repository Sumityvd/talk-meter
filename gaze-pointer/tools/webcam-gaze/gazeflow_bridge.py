"""Bridge: GazeFlow (GazeRecorder's GazePointer webcam tracker) -> MSREnable GazePointer / Windows cursor.

GazeFlow does the eye tracking, so accuracy is exactly GazeFlow's. This script reads its gaze through the
public GazeFlowAPI (https://github.com/szydej/GazeFlowAPI) and forwards it:
  --output udp     "x,y" physical pixels to GazePointer's Network sensor (dwell click, SnapRadius)
  --output print   just print samples (for checking the connection)

GazeFlowAPI protocol (from the published client GazeFlowAPI.cs):
  TCP 127.0.0.1:43333 -> send b"xml" -> send AppKey as a .NET BinaryWriter string (7-bit length prefix +
  UTF-8) -> read a status string starting with "ok" -> read one string per sample, an XML-serialized
  CGazeData (GazeX, GazeY, HeadX, HeadY, HeadZ, HeadYaw, HeadPitch, HeadRoll).
"AppKeyDemo" is the published test key; register your own at http://gazeflow.epizy.com/GazeFlowAPI/register/.

Usage (GazeFlow's GazePointer must be running and calibrated):
  py -3.12 gazeflow_bridge.py --output print
  py -3.12 gazeflow_bridge.py                      # forward to GazePointer on udp 4242
"""

import argparse
import socket
import sys
import time
import xml.etree.ElementTree as ElementTree

from webcam_gaze import UdpOutput, screen_info

FIELDS = ("GazeX", "GazeY", "HeadX", "HeadY", "HeadZ", "HeadYaw", "HeadPitch", "HeadRoll")


def dotnet_string(text):
    """BinaryWriter.Write(string): 7-bit encoded byte length, then UTF-8 bytes."""
    data = text.encode("utf-8")
    length, prefix = len(data), bytearray()
    while True:
        byte = length & 0x7F
        length >>= 7
        prefix.append(byte | (0x80 if length else 0))
        if not length:
            return bytes(prefix) + data


class GazeFlowClient:
    def __init__(self, host, port, app_key, timeout=5.0):
        self._socket = socket.create_connection((host, port), timeout=timeout)
        self._buffer = b""
        self._socket.sendall(b"xml")
        self._socket.sendall(dotnet_string(app_key))
        status = self._read_string()
        if not status.startswith("ok"):
            raise ConnectionError(f"GazeFlow refused the connection: {status!r}")
        self.status = status
        self._socket.settimeout(None)

    def _read_exact(self, count):
        while len(self._buffer) < count:
            chunk = self._socket.recv(65536)
            if not chunk:
                raise ConnectionError("GazeFlow closed the connection")
            self._buffer += chunk
        data, self._buffer = self._buffer[:count], self._buffer[count:]
        return data

    def _read_string(self):
        """BinaryReader.ReadString()."""
        length, shift = 0, 0
        while True:
            byte = self._read_exact(1)[0]
            length |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        return self._read_exact(length).decode("utf-8")

    def read(self):
        """Next sample as {field: float}."""
        text = self._read_string()
        try:
            root = ElementTree.fromstring(text.lstrip("﻿"))
        except ElementTree.ParseError:
            raise ConnectionError(f"unexpected message from GazeFlow: {text[:200]!r}")
        return {field: float(root.findtext(field, "nan")) for field in FIELDS}

    def close(self):
        self._socket.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gazeflow-port", type=int, default=43333, help="GazeFlowAPI port (GazeStreamSettings.xml)")
    parser.add_argument("--app-key", default="AppKeyDemo")
    parser.add_argument("--output", choices=("udp", "print"), default="udp")
    parser.add_argument("--port", type=int, default=4242, help="GazePointer NetworkPort (udp output)")
    parser.add_argument("--scale", default="auto",
                        help="factor from GazeFlow's coordinates to physical pixels: a number, or 'auto' "
                             "(detected from the first samples against the screen size)")
    args = parser.parse_args()

    width, height, _, _ = screen_info()   # also makes this process DPI aware (physical pixels)
    try:
        client = GazeFlowClient(args.host, args.gazeflow_port, args.app_key)
    except (OSError, ConnectionError) as error:
        sys.exit(f"cannot connect to GazeFlow on {args.host}:{args.gazeflow_port} ({error}). "
                 "Is GazeFlow's GazePointer running?")
    print(f"connected to GazeFlow: {client.status}", flush=True)

    output = UdpOutput(("127.0.0.1", args.port)) if args.output == "udp" else None
    scale = None if args.scale == "auto" else float(args.scale)
    seen_max = [0.0, 0.0]
    count, start = 0, time.monotonic()
    try:
        while True:
            sample = client.read()
            x, y = sample["GazeX"], sample["GazeY"]
            if x != x or y != y:   # NaN: no gaze (eyes closed / face lost)
                continue
            if scale is None:
                # GazeFlow is a DPI-unaware .NET app, so it may report logical pixels. Decide once gaze has
                # visited enough of the screen: coordinates never beyond the logical size => scale up.
                seen_max = [max(seen_max[0], x), max(seen_max[1], y)]
                count += 1
                if count >= 300:
                    dpi = ctypes_dpi_scale()
                    logical = (width / dpi, height / dpi)
                    scale = dpi if seen_max[0] <= logical[0] * 1.02 and seen_max[1] <= logical[1] * 1.02 else 1.0
                    print(f"coordinate scale {scale:g} (max seen {seen_max[0]:.0f},{seen_max[1]:.0f}; "
                          f"screen {width}x{height}, DPI scale {dpi:g})", flush=True)
            factor = scale or 1.0
            px, py = x * factor, y * factor
            if output:
                output((px, py))
            else:
                print(f"gaze {px:7.1f},{py:7.1f}  head x{sample['HeadX']:.1f} y{sample['HeadY']:.1f} "
                      f"z{sample['HeadZ']:.1f} yaw{sample['HeadYaw']:.1f} pitch{sample['HeadPitch']:.1f} "
                      f"roll{sample['HeadRoll']:.1f}", flush=True)
    except ConnectionError as error:
        print(error, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        if output:
            output.close()
        elapsed = time.monotonic() - start
        print(f"forwarded for {elapsed:.0f} s", flush=True)


def ctypes_dpi_scale():
    import ctypes
    return ctypes.windll.user32.GetDpiForSystem() / 96.0


if __name__ == "__main__":
    main()
