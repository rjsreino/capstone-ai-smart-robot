#!/bin/bash
# Install cloud server as systemd service

cd "$(dirname "$0")"

SERVICE_NAME="rovy-cloud.service"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME"

echo "================================"
echo "  INSTALL CLOUD SERVICE"
echo "================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "ERROR: Please run with sudo"
    echo "Usage: sudo ./install-service.sh"
    exit 1
fi

# Get current user (the one who ran sudo)
ACTUAL_USER="${SUDO_USER:-$USER}"
PROJECT_ROOT="$(cd .. && pwd)"

echo "Installing service for:"
echo "  User: $ACTUAL_USER"
echo "  Path: $PROJECT_ROOT"
echo ""

# Create service file
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=ROVY Cloud Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_ROOT
ExecStart=$PROJECT_ROOT/start_cloud.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

echo "✓ Service file created at $SERVICE_FILE"
echo ""

# Reload systemd
systemctl daemon-reload
echo "✓ Systemd reloaded"
echo ""

# Enable service
systemctl enable "$SERVICE_NAME"
echo "✓ Service enabled (will start on boot)"
echo ""

echo "================================"
echo "  INSTALLATION COMPLETE!"
echo "================================"
echo ""
echo "Service commands:"
echo "  Start:   sudo systemctl start $SERVICE_NAME"
echo "  Stop:    sudo systemctl stop $SERVICE_NAME"
echo "  Status:  sudo systemctl status $SERVICE_NAME"
echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "To start now:"
echo "  sudo systemctl start $SERVICE_NAME"
echo ""

