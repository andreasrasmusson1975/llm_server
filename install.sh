#!/usr/bin/env bash

# ============================================================================
# LLM Server Installation Script
# ============================================================================
#
# This script automates the complete installation and configuration of the
# LLM Server on a Ubuntu 22.04 system with NVIDIA GPU support.
#
# DESCRIPTION:
#   Installs and configures a production-ready LLM server with the following
#   components:
#   - CUDA Toolkit 12.1 for GPU acceleration
#   - Python virtual environment with PyTorch and Transformers
#   - FastAPI-based LLM server with Azure Key Vault integration
#   - Nginx reverse proxy with SSL/TLS termination
#   - Systemd service for automatic startup and management
#
# USAGE:
#   ./install.sh <DOMAIN_NAME> <EMAIL> <KEYVAULT_URL> <API_KEY_SECRET_NAME> <HUGGINGFACE_TOKEN>
#
# PARAMETERS:
#   DOMAIN_NAME           - Domain name for the server (e.g., api.example.com)
#   EMAIL                - Email address for Let's Encrypt certificate
#   KEYVAULT_URL         - Azure Key Vault URL for API key management
#   API_KEY_SECRET_NAME  - Secret name in Key Vault containing the API key
#   HUGGINGFACE_TOKEN    - HuggingFace access token for model downloads
#
# REQUIREMENTS:
#   - Ubuntu 22.04 LTS
#   - NVIDIA GPU with compatible drivers
#   - Sudo privileges
#   - Internet connection
#   - Domain name pointing to server IP
#
# WHAT IT INSTALLS:
#   - System packages (build tools, Python 3.10, etc.)
#   - NVIDIA CUDA Toolkit 12.1
#   - Python virtual environment at ~/llm_server/env
#   - PyTorch with CUDA support
#   - Transformers, FastAPI, and related dependencies
#   - Nginx with reverse proxy configuration
#   - Let's Encrypt SSL certificate via Certbot
#   - Systemd service for automatic startup
#
# FILES CREATED:
#   - /etc/llm_server.env          - System environment variables
#   - ~/.llm_server.env            - User environment variables
#   - ~/llm_server/env/            - Python virtual environment
#   - /etc/nginx/sites-*/llm_server - Nginx configuration
#   - /etc/systemd/system/llm_server.service - Systemd service
#
# AUTHOR: Andreas Rasmusson
# ============================================================================

set -euo pipefail

# -----------------------------
# Args and config
# -----------------------------
if [ $# -ne 5 ]; then
  echo "Usage: $0 <DOMAIN_NAME> <EMAIL> <KEYVAULT_URL> <API_KEY_SECRET_NAME> <HUGGINGFACE_TOKEN>"
  exit 1
fi

DOMAIN_NAME=$1
EMAIL=$2
KEYVAULT_URL=$3
API_KEY_SECRET_NAME=$4
HUGGINGFACE_TOKEN=$5
PYTHON_VERSION=3.10
VENV_DIR="$HOME/llm_server/env"
SERVICE_FILE="/etc/systemd/system/llm_server.service"

# -----------------------------
# Write Key Vault URL and secret name to env files
# -----------------------------
sudo tee /etc/llm_server.env > /dev/null <<EOF
KEYVAULT_URL=${KEYVAULT_URL}
API_KEY_SECRET_NAME=${API_KEY_SECRET_NAME}
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
EOF

tee ~/.llm_server.env > /dev/null <<EOF
export KEYVAULT_URL=${KEYVAULT_URL}
export API_KEY_SECRET_NAME=${API_KEY_SECRET_NAME}
export HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
EOF
echo 'source ~/.llm_server.env' >> ~/.bashrc
source ~/.bashrc
# Load env variables for current session
[ -f ~/.llm_server.env ] && source ~/.llm_server.env

# -----------------------------
# System packages
# -----------------------------
echo "📦 Installing system packages..."
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
    git wget curl unzip \
    build-essential ninja-build \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    pkg-config cmake \
    nginx gnupg software-properties-common ca-certificates apt-transport-https

echo "📦 Installing CUDA Toolkit 12.1..."
CUDA_REPO_PIN=cuda-ubuntu2204.pin
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${CUDA_REPO_PIN}
sudo mv ${CUDA_REPO_PIN} /etc/apt/preferences.d/cuda-repository-pin-600

# Add NVIDIA’s GPG key
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
  | sudo gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg

# Add NVIDIA repository (only once!)
if [ ! -f /etc/apt/sources.list.d/cuda.list ]; then
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] \
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
    | sudo tee /etc/apt/sources.list.d/cuda.list
fi

# Remove duplicate repo file if it exists
[ -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ] && sudo rm /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

sudo apt update
sudo apt install -y cuda-toolkit-12-1 cuda-compiler-12-1

# Ensure /usr/local/cuda points to the installed version
if [ -d /usr/local/cuda-12.1 ]; then
  sudo ln -sfn /usr/local/cuda-12.1 /usr/local/cuda
fi

# Export environment variables (safe for unset vars)
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib64' >> ~/.bashrc

# Apply immediately
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib64"


# Verify CUDA install
if ! command -v nvcc &> /dev/null; then
  echo "❌ nvcc not found after CUDA install"
  exit 1
else
  echo "✅ CUDA installed: $(nvcc -V | head -n 1)"
fi




echo "📦 Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "📦 Installing pip, wheel and setuptools in venv..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools

echo "📦 Installing llm_server (no deps) into venv..."
"$VENV_DIR/bin/python" -m pip install . --no-deps

echo "📦 Installing core dependencies into venv..."
"$VENV_DIR/bin/python" -m pip install "numpy<2"
"$VENV_DIR/bin/python" -m pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
"$VENV_DIR/bin/python" -m pip install transformers==4.56.1 accelerate==1.0.1 bitsandbytes==0.43.2
"$VENV_DIR/bin/python" -m pip install fastapi==0.111.0 uvicorn==0.30.1 azure-identity==1.15.0 azure-keyvault-secrets
"$VENV_DIR/bin/python" -m pip install packaging setuptools wheel ninja
"$VENV_DIR/bin/python" -m pip install flash-attn==2.6.1 --no-build-isolation


# -----------------------------
# Nginx reverse proxy for API
# -----------------------------
echo "🔧 Configuring Nginx as a reverse proxy..."
sudo mkdir -p /etc/nginx/locations

NGINX_CONF="/etc/nginx/sites-available/llm_server"
sudo tee $NGINX_CONF > /dev/null <<EOF
server {
    listen 80;
    server_name ${DOMAIN_NAME};
    include /etc/nginx/locations/*.conf;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
	proxy_buffering off;
    	proxy_cache off;
    	proxy_set_header Connection "";
    	chunked_transfer_encoding on;
	proxy_connect_timeout 600;
	proxy_send_timeout 600;
	proxy_read_timeout 600;
	send_timeout 600;
    }
}
EOF


sudo ln -sf $NGINX_CONF /etc/nginx/sites-enabled/llm_server
sudo nginx -t
sudo systemctl restart nginx
# -----------------------------
# Certbot for HTTPS
# -----------------------------
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d ${DOMAIN_NAME} --non-interactive --agree-tos -m ${EMAIL} || true

# -----------------------------
# Systemd service for llm-server
# -----------------------------
echo "🔧 Creating systemd service..."
sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=LLM Server (Uvicorn)
After=network.target

[Service]
User=${USER}
WorkingDirectory=${HOME}/llm_server
ExecStart=${VENV_DIR}/bin/uvicorn llm_server.api.server:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
EnvironmentFile=/etc/llm_server.env

[Install]
WantedBy=multi-user.target
EOF

echo "🔧 Setting up certbot auto-renew..."
sudo systemctl enable certbot.timer

sudo systemctl daemon-reload
sudo systemctl enable llm_server
sudo systemctl start llm_server

echo "✅ Installation complete."
echo "👉 Check service: sudo systemctl status llm_server"
echo "👉 Logs: journalctl -u llm_server -f"
echo "👉 Test API: https://${DOMAIN_NAME}/api/"
