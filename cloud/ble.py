"""
WiFi Manager - nmcli wrapper for WiFi operations
(Originally from ble.py, extracted for use without BLE)
"""
import subprocess
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class WifiConnectResult:
    success: bool
    message: str


class WifiManager:
    """
    Simple wrapper around `nmcli` to:
      - scan Wi-Fi networks
      - connect to Wi-Fi
      - get current connection status
    """

    def _run(self, args: List[str]) -> Tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)."""
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, err = proc.communicate()
        return proc.returncode, out.strip(), err.strip()

    def scan_networks(self) -> List[Dict]:
        """
        Returns a list of available Wi-Fi networks:
        [
          {"ssid": "NetworkName", "signal": 78, "security": "WPA2", "frequency": 5180},
          ...
        ]
        """
        # Force a rescan
        self._run(["nmcli", "dev", "wifi", "rescan"])

        code, out, err = self._run(
            ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY,FREQ", "dev", "wifi"]
        )
        if code != 0:
            raise RuntimeError(f"Wi-Fi scan failed: {err or out}")

        networks: List[Dict] = []
        seen = set()

        for line in out.splitlines():
            if not line:
                continue
            parts = line.split(":")
            if len(parts) < 4:
                continue
            
            ssid = parts[0].strip()
            signal_str = parts[1].strip()
            security = parts[2].strip()
            freq_str = parts[3].strip()
            
            if not ssid or ssid in seen:
                continue
            seen.add(ssid)
            
            try:
                signal = int(signal_str)
            except ValueError:
                signal = 0
            
            try:
                frequency = int(freq_str.replace(" MHz", ""))
            except ValueError:
                frequency = 0
            
            networks.append({
                "ssid": ssid,
                "signal": signal,
                "security": security if security else "Open",
                "frequency": frequency,
            })
        
        # Sort by signal strength
        networks.sort(key=lambda x: x["signal"], reverse=True)
        return networks

    def connect(self, ssid: str, password: str) -> WifiConnectResult:
        """
        Connect to a Wi-Fi network.
        Returns WifiConnectResult with success status and message.
        """
        if not ssid:
            return WifiConnectResult(False, "SSID cannot be empty")
        
        # Try to connect
        code, out, err = self._run([
            "nmcli", "dev", "wifi", "connect", ssid,
            "password", password
        ])
        
        if code == 0:
            return WifiConnectResult(True, f"Connected to {ssid}")
        else:
            return WifiConnectResult(False, err or out or "Connection failed")

    def current_connection(self) -> Tuple[bool, str, str]:
        """
        Get current WiFi connection status.
        Returns (connected: bool, network_name: str, ip_address: str)
        """
        # Check if connected
        code, out, err = self._run([
            "nmcli", "-t", "-f", "TYPE,STATE,CONNECTION", "dev", "status"
        ])
        
        connected = False
        network_name = ""
        
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0] == "wifi" and parts[1] == "connected":
                connected = True
                network_name = parts[2]
                break
        
        # Get IP address
        ip_address = ""
        if connected:
            code, out, err = self._run(["hostname", "-I"])
            if code == 0 and out:
                ip_address = out.split()[0]
        
        return connected, network_name, ip_address

    def disconnect(self) -> bool:
        """Disconnect from current WiFi network."""
        code, out, err = self._run(["nmcli", "dev", "disconnect", "wlan0"])
        return code == 0

