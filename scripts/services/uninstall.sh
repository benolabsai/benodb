#!/bin/bash
set -e

echo "=================================================="
echo "  BenoDB Search Background Service Uninstaller"
echo "=================================================="

# Detect OS
OS="$(uname -s)"
if [ "$OS" != "Linux" ] && [ "$OS" != "Darwin" ]; then
    echo "Error: Unsupported operating system: $OS"
    exit 1
fi

echo "Requesting administrative privileges for uninstallation..."
sudo -v

if [ "$OS" = "Linux" ]; then
    echo "Stopping and disabling systemd service..."
    sudo systemctl stop benodb-search.service || true
    sudo systemctl disable benodb-search.service || true
    
    echo "Removing systemd service file..."
    sudo rm -f /etc/systemd/system/benodb-search.service
    sudo systemctl daemon-reload
    
    CONF_DIR="/etc/benodb"
    
elif [ "$OS" = "Darwin" ]; then
    echo "Unloading and removing launchd service..."
    PLIST_DEST="/Library/LaunchDaemons/com.benodb.search.plist"
    sudo launchctl unload -w "$PLIST_DEST" 2>/dev/null || true
    sudo rm -f "$PLIST_DEST"
    
    echo "Removing macOS wrapper script..."
    sudo rm -f /usr/local/bin/benodb-search-runner.sh
    
    CONF_DIR="/usr/local/etc/benodb"
fi

echo "Removing binary..."
sudo rm -f /usr/local/bin/benodb-search

echo "Removing configuration files at $CONF_DIR..."
sudo rm -rf "$CONF_DIR"

echo "=================================================="
echo "  Uninstallation Complete!"
echo "=================================================="
