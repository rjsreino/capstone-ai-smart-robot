#!/usr/bin/env python3
import asyncio
import json
import socket
import subprocess
from typing import Dict, List, Tuple


def _dbus_return(signature: str):
    """Decorator to override the runtime return annotation for dbus-next."""

    def decorator(func):
        func.__annotations__["return"] = signature
        return func

    return decorator


from dbus_next import BusType, Variant
from dbus_next.aio import MessageBus
from dbus_next.errors import DBusError
from dbus_next.service import PropertyAccess, ServiceInterface, dbus_property, method

ROVY_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
WIFI_SCAN_UUID = "12345678-1234-5678-1234-56789abcdef1"
WIFI_CONFIG_UUID = "12345678-1234-5678-1234-56789abcdef2"
STATUS_UUID = "12345678-1234-5678-1234-56789abcdef3"

BLUEZ_SERVICE_NAME = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
ADAPTER_IFACE = "org.bluez.Adapter1"
APPLICATION_PATH = "/org/rovy/ble"


# ---------------------------------------------------------------------------
# WifiManager (from wifi_manager.py, adapted to live inside this file)
# ---------------------------------------------------------------------------


class WifiManager:
    """
    Simple wrapper around `nmcli` to:
      - scan Wi-Fi networks
      - connect to Wi-Fi
      - get current connection status
    Intended to be used from BLE GATT callbacks.
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
          {"ssid": "KT_GiGA_5G_1111", "signal": 78, "security": "WPA2", "frequency": 5180},
          {"ssid": "sejong-guest", "signal": 52, "security": "Open", "frequency": 2437},
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
            # Format: SSID:SIGNAL:SECURITY:FREQ
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
                frequency = int(freq_str)
            except ValueError:
                frequency = 0
            
            # Parse security type
            if not security or security == "--":
                security_type = "Open"
            else:
                security_type = security
            
            networks.append({
                "ssid": ssid,
                "signal": signal,
                "security": security_type,
                "frequency": frequency
            })

        # Sort by strongest signal
        networks.sort(key=lambda n: n["signal"], reverse=True)
        return networks

    def connect(self, ssid: str, password: str) -> Dict:
        """
        Connect to a Wi-Fi network.
        Returns:
          {
            "success": True/False,
            "message": "...",
          }
        """
        ssid = ssid.strip()
        if not ssid:
            return {"success": False, "message": "SSID cannot be empty"}

        # Add a connection or modify existing
        # For open APs you can pass an empty password.
        if password:
            args = [
                "nmcli",
                "dev",
                "wifi",
                "connect",
                ssid,
                "password",
                password,
            ]
        else:
            args = ["nmcli", "dev", "wifi", "connect", ssid]

        code, out, err = self._run(args)

        if code == 0:
            return {"success": True, "message": out or "Connected"}
        else:
            return {"success": False, "message": err or out or "Failed to connect"}

    def current_connection(self) -> Dict:
        """
        Returns info about current Wi-Fi connection if any.
        {
          "connected": bool,
          "network_name": str | None,
          "ip_address": str | None
        }
        """
        connected = False
        network_name = None
        ip_address = None

        # Try nmcli first (NetworkManager)
        try:
            result = subprocess.run(
                [
                    "nmcli",
                    "-t",
                    "-f",
                    "DEVICE,TYPE,STATE,CONNECTION",
                    "device",
                    "status",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split(":")
                    if len(parts) >= 4:
                        device = parts[0]
                        device_type = parts[1]
                        state = parts[2]
                        connection = parts[3]
                        # Check if it's a WiFi device and connected
                        if device_type == "wifi" and state == "connected" and connection:
                            network_name = connection
                            connected = True
                            # Get IP address for this device
                            ip_result = subprocess.run(
                                [
                                    "nmcli",
                                    "-t",
                                    "-f",
                                    "IP4.ADDRESS",
                                    "device",
                                    "show",
                                    device,
                                ],
                                capture_output=True,
                                text=True,
                                timeout=1,
                            )
                            if (
                                ip_result.returncode == 0
                                and ip_result.stdout.strip()
                            ):
                                output = ip_result.stdout.strip()
                                # Example: "IP4.ADDRESS[1]:192.168.200.123/24"
                                if ":" in output:
                                    ip_part = output.split(":", 1)[1]
                                    ip_address = ip_part.split("/")[0]
                                else:
                                    ip_address = output.split("/")[0]
                            break
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.SubprocessError,
        ) as exc:
            print(f"[ROVY BLE] nmcli not available or failed: {exc}")

        # If nmcli didn't work, try iwconfig
        if not connected:
            try:
                result = subprocess.run(
                    ["iwconfig"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "ESSID:" in line:
                            try:
                                essid_part = line.split("ESSID:")[1].strip()
                                if essid_part and essid_part != "off/any":
                                    network_name = essid_part.strip('"')
                                    connected = True
                            except (IndexError, ValueError):
                                pass
            except (
                subprocess.TimeoutExpired,
                FileNotFoundError,
                subprocess.SubprocessError,
            ) as exc:
                print(f"[ROVY BLE] iwconfig not available or failed: {exc}")

        # Get IP address from network interfaces if not already found
        if connected and not ip_address:
            try:
                # Try to get IP from common WiFi interfaces
                for interface in ["wlan0", "wlp2s0", "wlp3s0"]:
                    try:
                        result = subprocess.run(
                            ["ip", "-4", "addr", "show", interface],
                            capture_output=True,
                            text=True,
                            timeout=1,
                        )
                        if result.returncode == 0:
                            for line in result.stdout.split("\n"):
                                if "inet " in line:
                                    parts = line.strip().split()
                                    if len(parts) >= 2:
                                        ip_address = parts[1].split("/")[0]
                                        break
                            if ip_address:
                                break
                    except (
                        subprocess.TimeoutExpired,
                        subprocess.SubprocessError,
                    ):
                        continue
            except Exception as exc:
                print(
                    f"[ROVY BLE] Failed to get IP from network interfaces: {exc}"
                )

        # Fallback: try socket to get default route IP
        if connected and not ip_address:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip_address = s.getsockname()[0]
                s.close()
            except Exception as exc:
                print(f"[ROVY BLE] Failed to get IP via socket: {exc}")

        return {
            "connected": connected,
            "network_name": network_name,
            "ip_address": ip_address,
        }


# ---------------------------------------------------------------------------
# GATT boilerplate
# ---------------------------------------------------------------------------


class GattService(ServiceInterface):
    def __init__(self, index: int, uuid: str):
        super().__init__("org.bluez.GattService1")
        self.index = index
        self._uuid = uuid
        self._primary = True
        self.path = f"{APPLICATION_PATH}/service{index}"
        self.characteristics: List["GattCharacteristic"] = []

    @dbus_property(access=PropertyAccess.READ)
    @_dbus_return("s")
    def UUID(self) -> str:
        return self._uuid

    @dbus_property(access=PropertyAccess.READ)
    @_dbus_return("b")
    def Primary(self) -> bool:
        return self._primary

    @dbus_property(access=PropertyAccess.READ)
    @_dbus_return("ao")
    def Includes(self) -> List[str]:
        return []

    def add_characteristic(self, characteristic: "GattCharacteristic"):
        self.characteristics.append(characteristic)

    def export(self, bus: MessageBus):
        bus.export(self.path, self)
        for characteristic in self.characteristics:
            characteristic.export(bus)


class GattCharacteristic(ServiceInterface):
    def __init__(self, service: GattService, index: int, uuid: str, flags: List[str]):
        super().__init__("org.bluez.GattCharacteristic1")
        self.service = service
        self.index = index
        self._uuid = uuid
        self._flags = flags
        self.path = f"{service.path}/char{index}"

    def export(self, bus: MessageBus):
        bus.export(self.path, self)

    @dbus_property(access=PropertyAccess.READ)
    @_dbus_return("s")
    def UUID(self) -> str:
        return self._uuid

    @dbus_property(access=PropertyAccess.READ)
    @_dbus_return("o")
    def Service(self) -> str:
        return self.service.path

    @dbus_property(access=PropertyAccess.READ)
    @_dbus_return("as")
    def Flags(self) -> List[str]:
        return self._flags

    @dbus_property(access=PropertyAccess.READ)
    @_dbus_return("ao")
    def Descriptors(self) -> List[str]:
        return []

    @method()
    @_dbus_return("ay")
    def ReadValue(self, options: "a{sv}"):
        return self.read_value(options)

    @method()
    def WriteValue(self, value: "ay", options: "a{sv}"):
        self.write_value(value, options)

    def read_value(self, options):
        raise DBusError("org.bluez.Error.NotSupported", "Read not supported")

    def write_value(self, value, options):
        raise DBusError("org.bluez.Error.NotSupported", "Write not supported")


# ---------------------------------------------------------------------------
# Characteristics using WifiManager
# ---------------------------------------------------------------------------


class WifiScanCharacteristic(GattCharacteristic):
    def __init__(self, service: GattService, index: int, wifi_manager: WifiManager):
        super().__init__(service, index, WIFI_SCAN_UUID, ["read"])
        self.wifi_manager = wifi_manager

    def read_value(self, options):
        print("[ROVY BLE] WiFi scan requested")
        try:
            networks = self.wifi_manager.scan_networks()
            payload = json.dumps(networks)
            print(f"[ROVY BLE] Found {len(networks)} networks")
        except Exception as e:
            print(f"[ROVY BLE] WiFi scan error: {e}")
            payload = json.dumps({"error": str(e)})

        # dbus-next expects list of bytes
        return list(payload.encode("utf-8"))


class WifiConfigCharacteristic(GattCharacteristic):
    def __init__(self, service: GattService, index: int, wifi_manager: WifiManager):
        super().__init__(service, index, WIFI_CONFIG_UUID, ["write"])
        self.wifi_manager = wifi_manager
        self.last_status = "IDLE"
        self.last_message = ""

    def write_value(self, value, options):
        try:
            payload = bytes(value).decode("utf-8")
            data = json.loads(payload)
            ssid = data["ssid"]
            password = data.get("password", "")
        except (UnicodeDecodeError, KeyError, json.JSONDecodeError) as exc:
            self.last_status = "ERROR"
            self.last_message = f"Invalid payload: {exc}"
            print(f"[ROVY BLE] Invalid WiFi config data: {exc}")
            raise DBusError(
                "org.bluez.Error.InvalidValueLength", str(exc)
            ) from exc

        print(f"[ROVY BLE] Connecting to WiFi: {ssid}")
        result = self.wifi_manager.connect(ssid, password)
        self.last_status = "OK" if result.get("success") else "FAIL"
        self.last_message = result.get("message", "")

        if result.get("success"):
            print(f"[ROVY BLE] ✓ Connected to {ssid}: {self.last_message}")
        else:
            print(f"[ROVY BLE] ✗ Failed to connect to {ssid}: {self.last_message}")
            raise DBusError(
                "org.bluez.Error.Failed",
                self.last_message or "Failed to connect to Wi-Fi",
            )


class StatusCharacteristic(GattCharacteristic):
    def __init__(
        self,
        service: GattService,
        index: int,
        wifi_manager: WifiManager,
        config_char: WifiConfigCharacteristic,
    ):
        super().__init__(service, index, STATUS_UUID, ["read"])
        self.wifi_manager = wifi_manager
        self.config_char = config_char

    def read_value(self, options):
        conn = self.wifi_manager.current_connection()
        data = {
            "wifi_connected": conn.get("connected", False),
            "network_name": conn.get("network_name"),
            "ip_address": conn.get("ip_address"),
            "last_config_status": self.config_char.last_status,
            "last_config_message": self.config_char.last_message,
        }
        print(f"[ROVY BLE] Status requested: {data}")
        return list(json.dumps(data).encode("utf-8"))


# ---------------------------------------------------------------------------
# Gatt Application + main
# ---------------------------------------------------------------------------


class GattApplication(ServiceInterface):
    def __init__(self):
        super().__init__("org.freedesktop.DBus.ObjectManager")
        self.path = APPLICATION_PATH
        self.services: List[GattService] = []

    def add_service(self, service: GattService):
        self.services.append(service)

    def export(self, bus: MessageBus):
        bus.export(self.path, self)
        for service in self.services:
            service.export(bus)

    @method()
    @_dbus_return("a{oa{sa{sv}}}")
    def GetManagedObjects(self):
        managed: Dict[str, Dict[str, Dict[str, Variant]]] = {}
        for service in self.services:
            managed[service.path] = {
                "org.bluez.GattService1": {
                    "UUID": Variant("s", service._uuid),
                    "Primary": Variant("b", service._primary),
                    "Includes": Variant("ao", []),
                }
            }
            for char in service.characteristics:
                managed[char.path] = {
                    "org.bluez.GattCharacteristic1": {
                        "UUID": Variant("s", char._uuid),
                        "Service": Variant("o", char.service.path),
                        "Flags": Variant("as", char._flags),
                        "Descriptors": Variant("ao", []),
                    }
                }
        return managed


async def get_gatt_manager(bus: MessageBus, adapter_path: str):
    adapter_introspection = await bus.introspect(BLUEZ_SERVICE_NAME, adapter_path)
    adapter = bus.get_proxy_object(
        BLUEZ_SERVICE_NAME, adapter_path, adapter_introspection
    )
    return adapter.get_interface(GATT_MANAGER_IFACE)


async def find_adapter(bus: MessageBus) -> str:
    root_introspection = await bus.introspect(BLUEZ_SERVICE_NAME, "/")
    obj = bus.get_proxy_object(BLUEZ_SERVICE_NAME, "/", root_introspection)
    manager = obj.get_interface("org.freedesktop.DBus.ObjectManager")
    managed_objects = await manager.call_get_managed_objects()
    for path, ifaces in managed_objects.items():
        if ADAPTER_IFACE in ifaces:
            return path
    raise RuntimeError("No Bluetooth adapter found")


async def main():
    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    print("[ROVY BLE] Starting GATT Server...")

    wifi_manager = WifiManager()

    app = GattApplication()

    wifi_service = GattService(0, ROVY_SERVICE_UUID)
    app.add_service(wifi_service)

    wifi_scan_char = WifiScanCharacteristic(wifi_service, 0, wifi_manager)
    wifi_config_char = WifiConfigCharacteristic(wifi_service, 1, wifi_manager)
    status_char = StatusCharacteristic(
        wifi_service, 2, wifi_manager, wifi_config_char
    )

    wifi_service.add_characteristic(wifi_scan_char)
    wifi_service.add_characteristic(wifi_config_char)
    wifi_service.add_characteristic(status_char)

    app.export(bus)

    gatt_manager = None
    try:
        adapter_path = await find_adapter(bus)
        gatt_manager = await get_gatt_manager(bus, adapter_path)
        await gatt_manager.call_register_application(app.path, {})
        print(f"[ROVY BLE] ✓ GATT application registered on {adapter_path}")
        print(f"[ROVY BLE] Service UUID: {ROVY_SERVICE_UUID}")
        print(f"[ROVY BLE] Device name: ROVY")
        print(f"[ROVY BLE] Ready for connections!")

        while True:
            await asyncio.sleep(1)

    except Exception as exc:
        print(f"[ROVY BLE] ✗ Error initializing GATT application: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        if gatt_manager is not None:
            try:
                await gatt_manager.call_unregister_application(app.path)
                print("[ROVY BLE] Unregistered GATT application")
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())