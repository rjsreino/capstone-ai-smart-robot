#!/bin/bash
# Script to check if depth_llava_nav.py is running

echo "=== Checking Automatic Navigation Status ==="
echo ""

# Check for running processes
echo "1. Running processes:"
ps aux | grep -E "python.*depth_llava_nav" | grep -v grep || echo "   No depth_llava_nav.py process found"
echo ""

# Check for zombie processes
echo "2. Zombie/defunct processes:"
ps aux | grep -E "depth_llava|autonav" | grep -E "defunct|Z" | grep -v grep || echo "   No zombie processes found"
echo ""

# Check log files
echo "3. Log files:"
if [ -d "/home/jetson/rovy/logs" ]; then
    if [ -f "/home/jetson/rovy/logs/autonav_stdout.log" ]; then
        echo "   stdout.log exists (last 10 lines):"
        tail -10 /home/jetson/rovy/logs/autonav_stdout.log | sed 's/^/   /'
    else
        echo "   stdout.log not found"
    fi
    echo ""
    if [ -f "/home/jetson/rovy/logs/autonav_stderr.log" ]; then
        echo "   stderr.log exists (last 10 lines):"
        tail -10 /home/jetson/rovy/logs/autonav_stderr.log | sed 's/^/   /'
    else
        echo "   stderr.log not found"
    fi
else
    echo "   Logs directory doesn't exist"
fi
echo ""

# Check service status
echo "4. Rovy Assistant Service:"
systemctl is-active rovy-assistant && echo "   Service is active" || echo "   Service is not active"
echo ""

# Check recent journal logs
echo "5. Recent AutoNav logs from journal:"
journalctl -u rovy-assistant --since "10 minutes ago" | grep -i "autonav" | tail -5 | sed 's/^/   /' || echo "   No recent AutoNav logs"

