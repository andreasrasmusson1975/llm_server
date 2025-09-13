#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="env"

echo "📦 Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source env/bin/activate

echo "📦 Upgrading pip in venv..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip

echo "📦 Installing llm_server (no deps) into venv..."
"$VENV_DIR/bin/python" -m pip install . --no-deps

echo "📦 Installing core dependencies into venv..."
"$VENV_DIR/bin/python" -m pip install "numpy<2"
"$VENV_DIR/bin/python" -m pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --extra-inde>"$VENV_DIR/bin/python" -m pip install transformers==4.56.1 accelerate==1.0.1 bitsandbytes==0.43.2
"$VENV_DIR/bin/python" -m pip install fastapi==0.111.0 uvicorn==0.30.1 azure-identity==1.15.0 azure-keyvault-secrets==4>
echo "✅ Installation complete."
