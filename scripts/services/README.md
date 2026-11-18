# BenoDB Search Background Services

This directory contains configuration files and an installer script to run `benodb-search` natively in the background, allowing it to start automatically on system boot.

Running natively provides the best performance and enables direct GPU (CUDA/wgpu) access without needing to configure complex Docker GPU passthrough.

By default, the Qdrant-compatible REST API is exposed on `http://localhost:6333`.

## Automated Installation (Linux & macOS)

### Method 1: Using the Python CLI (Recommended)

If you have installed the `benodb` Python package via PyPI, you can install and configure the background service directly from the terminal:

```bash
sudo benodb install-service
```
This automatically detects your OS and installs the required `systemd` or `launchd` configuration files. (Note: you must have the `benodb-search` binary built or installed at `/usr/local/bin/benodb-search`, or pass `--binary-path`).

### Method 2: Using the Shell Scripts

If you are building from source and prefer not to use the Python CLI:

Ensure you have built the `benodb-search` binary first:
```bash
cargo build --release -p benodb-search
```

Then run the installer:
```bash
./install.sh
```

The script will automatically detect your OS, install the binary to `/usr/local/bin`, and configure the background service (`systemd` for Linux, `launchd` for macOS).

## Configuration

The background service uses a centralized configuration file where you can adjust environment variables (such as enabling GPU, changing ports, or configuring storage).

- **Linux**: Edit `/etc/benodb/benodb-search.conf`
- **macOS**: Edit `/usr/local/etc/benodb/benodb-search.conf`

After changing the configuration file, you must restart the service:

- **Linux**:
  ```bash
  sudo systemctl restart benodb-search.service
  ```
- **macOS**:
  ```bash
  sudo launchctl unload -w /Library/LaunchDaemons/com.benodb.search.plist
  sudo launchctl load -w /Library/LaunchDaemons/com.benodb.search.plist
  ```

## Viewing Logs

- **Linux**:
  ```bash
  sudo journalctl -u benodb-search.service -f
  ```
- **macOS**:
  ```bash
  tail -f /tmp/benodb-search.log
  ```

## Uninstallation

To completely remove the service, configuration, and binary, you can use the Python CLI:
```bash
sudo benodb uninstall-service
```

Or you can use the included script:
```bash
./uninstall.sh
```
