import threading
import time
from rover_controller import Rover


base = Rover('/dev/ttyACM0', 115200)

# Breathing light disabled - lights stay off on startup
# threading.Thread(target=lambda: base.breath_light(20), daemon=True).start()

base.display_text(0, "ROVY")
base.display_text(1, f"sbc_version: 0.93")
base.display_text(2, f"{20}")
base.display_text(3, "Starting...")

import os_info

si = os_info.SystemInfo()


def update_data_loop():
    while 1:
        wlan = si.wlan_ip
        ssid = si.wifi_name
        
        try:
            status = base.get_status()
            voltage = status.get('voltage', 0.0)
            battery_pct = base._voltage_to_percentage(voltage)
            battery_info = f"Bat:{battery_pct}%"
        except:
            battery_info = "Bat:---%"
        
        if ssid:
            base.display_text(0, f"W:{ssid} {si.wifi_rssi}dBm {battery_info}")
        else:
            base.display_text(0, f"W: NO {si.net_interface} {battery_info}")
        if wlan:
            base.display_text(1, f"IP:{ssid}")
        else:
            base.display_text(1, f"IP: NO {si.net_interface}")
        time.sleep(5)
        
# commandline on boot
def cmd_on_boot():
    cmd_list = [
        'base -c {"T":142,"cmd":50}',   # set feedback interval
        'base -c {"T":131,"cmd":1}',    # serial feedback flow on
        'base -c {"T":143,"cmd":0}',    # serial echo off
        'base -c {{"T":4,"cmd":{}}}'.format(2),      # select the module - 0:None 1:RoArm-M2-S 2:Gimbal
        'base -c {"T":300,"mode":0,"mac":"EF:EF:EF:EF:EF:EF"}',  # the base won't be ctrl by esp-now broadcast cmd, but it can still recv broadcast megs.
        'send -a -b'    # add broadcast mac addr to peer
    ]
    print('base -c {{"T":4,"cmd":{}}}'.format(2))

if __name__ == "__main__":
    # breath light off
    #base.change_breath_light_flag(False)
    
    # lights off
    # base.lights_ctrl(255, 255)
    
    # pt/arm looks forward
    base.gimbal_ctrl(0, 0, 200, 10)
    
    # lights off
    # base.lights_ctrl(0, 0)
    
    # feedback loop starts
    si.start()
    si.resume()
    data_update_thread = threading.Thread(target=update_data_loop, daemon=True)
    data_update_thread.start()
    
    
    cmd_on_boot()