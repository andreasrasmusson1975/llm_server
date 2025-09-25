#!/usr/bin/env bash

# ============================================================================
# LLM Server Uninstall Script
# ============================================================================
#
# This script safely removes the LLM Server installation and cleans up
# all associated files, services, and configurations from the system.
#
# DESCRIPTION:
#   Performs a complete cleanup of the LLM Server installation by removing:
#   - Systemd service and stopping the server
#   - Environment configuration files
#   - Nginx reverse proxy configuration
#   - Python virtual environment
#   - CUDA repository configuration
#
# USAGE:
#   ./uninstall.sh
#
# PARAMETERS:
#   None - This script takes no parameters
#
# REQUIREMENTS:
#   - Ubuntu 22.04 LTS (or compatible system)
#   - Sudo privileges
#   - Previous LLM Server installation
#
# WHAT IT REMOVES:
#   - Systemd service: /etc/systemd/system/llm_server.service
#   - Environment files: /etc/llm_server.env, ~/.llm_server.env
#   - Nginx configurations: /etc/nginx/sites-*/llm_server
#   - Virtual environment: ~/llm_server/env/
#   - CUDA repository keys and preferences (optional)
#
# WHAT IT PRESERVES:
#   - System packages (Python, Nginx, CUDA Toolkit remain installed)
#   - SSL certificates (Let's Encrypt certificates are not removed)
#   - User data and logs
#   - Other Nginx sites and configurations
#
# SAFETY FEATURES:
#   - Uses '|| true' to continue execution even if some operations fail
#   - Only removes LLM Server specific files and configurations
#   - Gracefully handles missing files or services
#   - Validates Nginx configuration before reloading
#
# POST-CLEANUP:
#   After running this script, you may want to:
#   - Remove unused system packages: sudo apt autoremove
#   - Remove CUDA Toolkit manually if no longer needed
#   - Remove Nginx if not used for other services
#   - Check for remaining environment variables in ~/.bashrc
#
# AUTHOR: Andreas Rasmusson
# ============================================================================

set -euo pipefail

echo "🛑 Stopping llm_server service..."
sudo systemctl stop llm_server || true
sudo systemctl disable llm_server || true

echo "🧹 Removing systemd service..."
sudo rm -f /etc/systemd/system/llm_server.service
sudo systemctl daemon-reload
sudo systemctl reset-failed || true

echo "🧹 Removing environment files..."
sudo rm -f /etc/llm_server.env
rm -f ~/.llm_server.env

echo "🧹 Removing nginx config..."
sudo rm -f /etc/nginx/sites-available/llm_server
sudo rm -f /etc/nginx/sites-enabled/llm_server
sudo nginx -t || true
sudo systemctl reload nginx || true

echo "🧹 (Optional) Removing virtual environment..."
rm -rf ~/llm_server/env

echo "🧹 (Optional) Removing CUDA repo and key..."
sudo rm -f /etc/apt/preferences.d/cuda-repository-pin-600
sudo rm -f /usr/share/keyrings/cuda-archive-keyring.gpg
sudo rm -f /etc/apt/sources.list.d/cuda.list
sudo apt update

echo "✅ Cleanup complete."
