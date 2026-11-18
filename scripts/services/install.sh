#!/bin/bash
set -e

echo "=================================================="
echo "  BenoDB Search Background Service Installer"
echo "=================================================="

# Detect OS
OS="$(uname -s)"
if [ "$OS" != "Linux" ] && [ "$OS" != "Darwin" ]; then
    echo "Error: Unsupported operating system: $OS"
    exit 1
fi

# Ensure benodb-search is built or available in target/release
BINARY_SRC="../../target/release/benodb-search"
if [ ! -f "$BINARY_SRC" ]; then
    echo "Warning: $BINARY_SRC not found."
    echo "Make sure you build the project first with: cargo build --release -p benodb-search"
    # We won't exit here, just warn them, in case they already have it installed
fi

# Need sudo for installation
echo "Requesting administrative privileges for installation..."
sudo -v

echo "Installing benodb-search binary to /usr/local/bin..."
if [ -f "$BINARY_SRC" ]; then
    sudo cp "$BINARY_SRC" /usr/local/bin/benodb-search
fi
sudo chmod +x /usr/local/bin/benodb-search

# Install configuration file
if [ "$OS" = "Linux" ]; then
    CONF_DIR="/etc/benodb"
else
    # macOS
    CONF_DIR="/usr/local/etc/benodb"
fi

echo "Creating configuration directory at $CONF_DIR..."
sudo mkdir -p "$CONF_DIR"

echo "Installing benodb-search.conf..."
sudo cp benodb-search.conf "$CONF_DIR/"
echo "You can configure your settings by editing: $CONF_DIR/benodb-search.conf"

# Install Services
if [ "$OS" = "Linux" ]; then
    echo "Installing systemd service (Linux)..."
    sudo cp benodb-search.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable benodb-search.service
    sudo systemctl start benodb-search.service
    echo "Service installed and started! Check logs with: sudo journalctl -u benodb-search.service -f"
    
elif [ "$OS" = "Darwin" ]; then
    echo "Installing launchd service (macOS)..."
    sudo cp benodb-search-runner.sh /usr/local/bin/
    sudo chmod +x /usr/local/bin/benodb-search-runner.sh
    
    PLIST_DEST="/Library/LaunchDaemons/com.benodb.search.plist"
    sudo cp com.benodb.search.plist "$PLIST_DEST"
    sudo chown root:wheel "$PLIST_DEST"
    
    # Reload if it was already loaded
    sudo launchctl unload -w "$PLIST_DEST" 2>/dev/null || true
    sudo launchctl load -w "$PLIST_DEST"
    
    echo "Service installed and started! Check logs with: tail -f /tmp/benodb-search.log"
fi

echo "=================================================="
echo "  Installation Complete!"
echo "=================================================="
