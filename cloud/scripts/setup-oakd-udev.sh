#!/usr/bin/env bash
# Setup script for OAK-D camera udev rules
# This allows non-root users to access the OAK-D device

set -euo pipefail

UDEV_RULES_FILE="/etc/udev/rules.d/80-movidius.rules"

echo "Setting up OAK-D (Luxonis) udev rules..."

# Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)" 
   exit 1
fi

# Create udev rules file
cat > "$UDEV_RULES_FILE" <<'EOF'
# Luxonis OAK-D devices
SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"
EOF

echo "✓ Created $UDEV_RULES_FILE"

# Reload udev rules
udevadm control --reload-rules
udevadm trigger

echo "✓ Reloaded udev rules"
echo ""
echo "OAK-D setup complete! The device should now be accessible without root privileges."
echo "You may need to unplug and replug the OAK-D device for changes to take effect."

