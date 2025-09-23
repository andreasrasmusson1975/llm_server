"""
# Install/Uninstall Package

## Installation Instructions

The `llm_server` package includes automated installation and uninstallation scripts for Linux systems. These scripts handle the complete setup of the LLM server including system dependencies, CUDA toolkit, Python environment, and production deployment configuration with Nginx and systemd.

### Prerequisites

- Ubuntu 22.04 LTS (recommended)
- NVIDIA GPU with CUDA support
- Root/sudo access
- Domain name pointing to the server (for HTTPS)
- Azure Key Vault with API key secret

### Installation Script Usage

The `install.sh` script performs a complete production installation of the LLM server:

```bash
# Make the script executable
chmod +x install.sh

# Run installation with required parameters
./install.sh <DOMAIN_NAME> <EMAIL> <KEYVAULT_URL> <API_KEY_SECRET_NAME> <HUGGINGFACE_TOKEN>
```

**Parameters:**
- `DOMAIN_NAME`: Your domain name (e.g., api.example.com)
- `EMAIL`: Email address for Let's Encrypt certificate
- `KEYVAULT_URL`: Azure Key Vault URL (e.g., https://your-vault.vault.azure.net/)
- `API_KEY_SECRET_NAME`: Name of the API key secret in Key Vault
- `HUGGINGFACE_TOKEN`: Hugging Face API token for model downloads

**Example:**
```bash
./install.sh api.mycompany.com admin@mycompany.com https://llm-vault.vault.azure.net/ llm-api-key hf_your_token_here
```

### What the Installation Script Does

1. **System Package Installation**:
   - Updates Ubuntu packages
   - Installs build tools, Python 3.10, development headers
   - Installs Nginx web server

2. **CUDA Toolkit Setup**:
   - Downloads and installs CUDA Toolkit 12.1
   - Configures CUDA environment variables
   - Verifies CUDA installation

3. **Python Environment**:
   - Creates virtual environment at `~/llm_server/env`
   - Installs PyTorch with CUDA support
   - Installs Transformers, FastAPI, and other dependencies
   - Installs Flash Attention 2 for performance

4. **LLM Server Installation**:
   - Installs the `llm_server` package
   - Creates environment configuration files

5. **Production Deployment**:
   - Configures Nginx reverse proxy
   - Sets up SSL certificates with Let's Encrypt
   - Creates systemd service for automatic startup
   - Starts the LLM server service

6. **Environment Configuration**:
   - Creates `/etc/llm_server.env` with Azure Key Vault settings
   - Creates `~/.llm_server.env` for user access

### Post-Installation

After successful installation:

```bash
# Check service status
sudo systemctl status llm_server

# View logs
journalctl -u llm_server -f

# Test the API
curl "https://your-domain.com/api/ping?x_api_key=your-key"
```

The API will be available at:
- HTTP: `http://your-domain.com/api/`
- HTTPS: `https://your-domain.com/api/`

### Uninstallation Script Usage

The `uninstall.sh` script removes the LLM server and cleans up system modifications:

```bash
# Make the script executable
chmod +x uninstall.sh

# Run uninstallation
./uninstall.sh
```

### What the Uninstallation Script Does

1. **Service Cleanup**:
   - Stops and disables the llm_server systemd service
   - Removes the systemd service file
   - Reloads systemd daemon

2. **Configuration Cleanup**:
   - Removes environment files (`/etc/llm_server.env`, `~/.llm_server.env`)
   - Removes Nginx configuration files
   - Reloads Nginx configuration

3. **Optional Cleanup**:
   - Removes Python virtual environment
   - Removes CUDA repository configuration
   - Updates package lists

**Note**: The uninstall script does not remove:
- System packages (Python, Nginx, CUDA)
- SSL certificates
- The main project directory

### Manual Installation Alternative

For development or custom deployments, you can install manually:

```bash
# Create virtual environment
python3 -m venv ~/llm_server/env
source ~/llm_server/env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install the package
pip install .

# Set environment variables
export KEYVAULT_URL="https://your-vault.vault.azure.net/"
export API_KEY_SECRET_NAME="your-secret-name"

# Start the server
start_llm_server
```

### Troubleshooting

**CUDA Issues**:
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Check CUDA environment
echo $CUDA_HOME
echo $PATH
```

**Service Issues**:
```bash
# Check service status
sudo systemctl status llm_server

# View detailed logs
journalctl -u llm_server --no-pager

# Restart service
sudo systemctl restart llm_server
```

**API Issues**:
```bash
# Test local connection
curl "http://127.0.0.1:8000/ping?x_api_key=your-key"

# Check Nginx configuration
sudo nginx -t
sudo systemctl status nginx
```
"""