"""
# Helper Functionality Package

## Introduction

The `helper_functionality` package provides utility functions for model initialization and configuration management in the LLM server. It handles the setup of Hugging Face transformers models with performance optimizations and centralizes YAML-based configuration loading across the application.

## Core Components

- **Model Loading**: Initialization and optimization of Hugging Face language models
- **YAML Configuration**: Centralized loading and parsing of configuration files
- **Performance Optimization**: Model compilation and quantization setup
- **Configuration Management**: Type-safe access to model, server, and Azure settings

## Features

- **Optimized Model Loading**: Automatic device mapping, Flash Attention 2, and torch compilation
- **4-bit Quantization Support**: Memory-efficient model loading with BitsAndBytes (configurable)
- **YAML Configuration Loading**: Safe parsing of configuration files with structured access
- **Multiple Configuration Types**: Support for model, server, Azure, and system prompt configurations
- **Path Resolution**: Automatic resolution of configuration file paths relative to project structure
- **Type Safety**: Structured return types for configuration data

## Technical Architecture

The package is organized into two main modules:

- **model_loading.py**: Handles the initialization of Hugging Face models with performance optimizations including Flash Attention 2, torch compilation, and optional quantization
- **yaml_loading.py**: Provides a centralized interface for loading and accessing YAML configuration files with automatic path resolution

Configuration files are stored in the `yaml_files` directory and accessed through dedicated loader functions that return parsed dictionaries with predictable structure.

## Usage

### Model Initialization

```python
from llm_server.helper_functionality.model_loading import initialize

# Load model with automatic device mapping and optimizations
tokenizer, model = initialize(
    device="auto",  # or "cuda", "cpu", specific GPU index
    model_id="meta-llama/Llama-3.1-8B-Instruct"
)

# Model is returned ready for inference with:
# - Flash Attention 2 implementation
# - Torch compilation for performance
# - High precision matrix multiplication
# - bfloat16 precision
```

### Configuration Loading

```python
from llm_server.helper_functionality.yaml_loading import (
    load_model_config,
    load_system_prompt,
    load_chat_server_config,
    load_azure_config
)

# Load model configuration
model_config = load_model_config()
model_id = model_config.get("id")
device = model_config.get("device")
parameters = model_config.get("parameters", {})

# Load system prompt
system_prompt = load_system_prompt()

# Load server configuration
server_config = load_chat_server_config()
host = server_config.get("host")
port = server_config.get("port")

# Load Azure configuration
azure_config = load_azure_config()
vault_url = azure_config.get("key_vault_url")
secret_name = azure_config.get("api_key_secret_name")
```

### Custom YAML Loading

```python
from llm_server.helper_functionality.yaml_loading import load_yaml

# Load any YAML file from the yaml_files directory
custom_config = load_yaml("custom_config.yaml")

# Safe loading prevents code execution in YAML files
data = custom_config.get("custom_section", {})
```

### Complete Initialization Example

```python
from llm_server.helper_functionality.model_loading import initialize
from llm_server.helper_functionality.yaml_loading import (
    load_model_config, 
    load_system_prompt
)

# Load configuration
config = load_model_config()
system_prompt = load_system_prompt()

# Initialize model with config values
tokenizer, model = initialize(
    device=config.get("device", "auto"),
    model_id=config.get("id")
)

# Ready for inference
print(f"Model loaded: {config.get('id')}")
print(f"Device: {config.get('device')}")
print(f"System prompt loaded: {len(system_prompt)} characters")
```

### Configuration File Structure

Expected YAML file formats:

```yaml
# model_config.yaml
model:
  id: "meta-llama/Llama-3.1-8B-Instruct"
  device: "auto"
  parameters:
    max_new_tokens: 4096
    temperature: 0.7

# server_config.yaml  
server:
  host: "0.0.0.0"
  port: 8000
  api_key: "your-api-key"

# azure_config.yaml
azure:
  key_vault_url: "https://your-vault.vault.azure.net/"
  api_key_secret_name: "api-key-secret"
```
"""