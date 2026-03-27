import os, psutil, time
import subprocess, re, netifaces
import threading
from jtop import jtop
from jtop.core.exceptions import JtopException  # <-- import the exception

curpath = os.path.realpath(__file__)
thisPath = os.path.dirname(curpath)


class SystemInfo(threading.Thread):
    """docstring for SystemInfo"""
    def __init__(self):
        self.this_path = None

        self.pictures_size = 0
        self.videos_size = 0
        self.cpu_load = 0
        self.cpu_temp = 0
        self.ram = 0
        self.wifi_rssi = 0

        self.net_interface = "wlan0"
        self.wlan_ip = "None"
        self.eth0_ip = "None"
        self.wifi_mode = "None"
        self.wifi_name = "None"

        self.update_interval = 2

        # flag so we don't keep trying jtop if it fails once
        self._jtop_enabled = True

        super(SystemInfo, self).__init__()
        self.__flag = threading.Event()
        self.__flag.clear()

    def get_wifi_name(self):
        """Return connected Wi-Fi SSID"""
        try:
            ssid = subprocess.check_output(
                ["iwgetid", "-r"], stderr=subprocess.DEVNULL
            ).decode().strip()
            return ssid if ssid else "None"
        except Exception:
            return "None"

    def get_folder_size(self, folder_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return round(total_size / (1024 * 1024), 2)

    def update_folder_size(self):
        self.pictures_size = self.get_folder_size(self.this_path + '/templates/pictures')
        self.videos_size = self.get_folder_size(self.this_path + '/templates/videos')

    def update_folder(self, input_path):
        self.this_path = input_path
        threading.Thread(target=self.update_folder_size, daemon=True).start()

    def get_info_jtop(self):
        if not self._jtop_enabled:
            return

        try:
            with jtop() as jetson:
                if jetson.ok():
                    self.cpu_temp = round(jetson.stats['Temp cpu'], 2)
                    self.ram = round(
                        jetson.memory['RAM']['used'] / jetson.memory['RAM']['tot'] * 100, 2
                    )
                    self.cpu_load = jetson.stats['CPU1']
        except JtopException as e:
            # jtop.service not running or unavailable: log once and disable
            print("[SystemInfo] jtop not available:", e)
            self._jtop_enabled = False
        except Exception as e:
            # Any other unexpected error: log & disable to avoid crashing the thread
            print("[SystemInfo] Unexpected jtop error:", e)
            self._jtop_enabled = False

    def get_ip_address(self, interface):
        try:
            interface_info = netifaces.ifaddresses(interface)
            ipv4_info = interface_info.get(netifaces.AF_INET, [{}])
            return ipv4_info[0].get('addr', None)
        except Exception:
            return None

    def get_wifi_mode(self):
        if self.wlan_ip == '192.168.50.5':
            return "AP"
        else:
            return "STA"

    def get_signal_strength(self):
        return 0  # You can fill later

    def change_net_interface(self, new_interface):
        self.net_interface = new_interface

    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def run(self):
        self.eth0_ip = self.get_ip_address('eth0')
        self.wlan_ip = self.get_ip_address(self.net_interface)
        self.wifi_mode = self.get_wifi_mode()
        self.wifi_rssi = self.get_signal_strength()
        self.wifi_name = self.get_wifi_name()

        while True:
            self.get_info_jtop()
            self.wifi_rssi = self.get_signal_strength()
            time.sleep(0.5)
            self.wifi_mode = self.get_wifi_mode()
            time.sleep(0.5)
            self.wlan_ip = self.get_ip_address(self.net_interface)
            time.sleep(0.5)
            self.eth0_ip = self.get_ip_address('eth0')
            time.sleep(0.5)
            self.wifi_name = self.get_wifi_name()
            self.__flag.wait()
