"""
# YAML Files Package

## Introduction

The `yaml_files` package contains structured configuration files that define the behavior and settings for the LLM server application. These YAML files serve as the central configuration store for model parameters, server settings, Azure integration, and system prompts used throughout the application.

## Core Components

- **model_config.yaml**: Language model configuration and generation parameters
- **server_config.yaml**: HTTP server settings including host, port, and authentication
- **azure_config.yaml**: Azure Key Vault integration settings for secure credential management
- **system_prompt_config.yaml**: System prompts for different AI assistant roles and behaviors

## Features

- **Centralized Configuration**: Single location for all application settings
- **Environment Variable Integration**: Support for environment variable placeholders
- **Role-Based System Prompts**: Multiple prompt configurations for different assistant behaviors
- **Model Parameter Control**: Fine-grained control over language model generation settings
- **Secure Credential Management**: Azure Key Vault integration for sensitive data
- **Human-Readable Format**: YAML structure for easy editing and version control

## Technical Architecture

The YAML files are organized by functional area and loaded through the `helper_functionality.yaml_loading` module. Each file follows a consistent structure with top-level keys organizing related configuration sections. Environment variable placeholders are used for sensitive or deployment-specific values.

The configuration system supports:
- Safe YAML loading to prevent code execution
- Automatic path resolution relative to the package location
- Structured access through dedicated loader functions
- Default value handling for missing configuration keys

## Usage

### Model Configuration

```yaml
# model_config.yaml
model:
  type: "llm"
  id: "meta-llama/Llama-3.1-8B-Instruct"
  device: "auto"
  parameters:
    max_new_tokens: 4096
    temperature: 0.7
    top_p: 0.9
    do_sample: True
    use_cache: True
    return_dict_in_generate: True
```

### Server Configuration

```yaml
# server_config.yaml
server:
  host: "CHAT_SERVER_HOST"  # Environment variable placeholder
  port: 8000
  api_key: "CHAT_SERVER_API_KEY"  # Environment variable placeholder
```

### Azure Integration

```yaml
# azure_config.yaml
azure:
  key_vault_url: "KEYVAULT_URL"  # Environment variable placeholder
  api_key_secret_name: "API_KEY_SECRET_NAME"  # Environment variable placeholder
```

### System Prompts

```yaml
# system_prompt_config.yaml
system_prompts:
  assistant: |
    # General-Purpose Assistant — System Prompt
    
    ## Identity & Mission
    You are a helpful, knowledgeable, and reliable general-purpose assistant.
    Your job is to solve the user's task as directly as possible,
    with accurate information, clear structure, and minimal fluff.
    
    ## Core Priorities (in order)
    1) Correctness & safety
    2) Following the user's instructions and constraints
    3) Clarity and usefulness of the output
    
  reviser: |
    # Reviser — System Prompt
    
    ## Identity & Mission
    You are an expert at taking the user's drafts and improving them.
    Your job is to make draft answers better by including more relevant facts,
    correcting errors, and structuring the revised answer clearly.
```

### Loading Configuration in Code

```python
from llm_server.helper_functionality.yaml_loading import (
    load_model_config,
    load_chat_server_config,
    load_azure_config,
    load_system_prompt
)

# Load specific configurations
model_config = load_model_config()
server_config = load_chat_server_config()
azure_config = load_azure_config()
system_prompt = load_system_prompt()

# Access configuration values
model_id = model_config.get("id")
temperature = model_config.get("parameters", {}).get("temperature", 0.7)
server_port = server_config.get("port", 8000)
vault_url = azure_config.get("key_vault_url")
```

### Environment Variable Setup

Set environment variables for deployment-specific values:

```bash
# Windows PowerShell
$env:CHAT_SERVER_HOST = "0.0.0.0"
$env:CHAT_SERVER_API_KEY = "your-secure-api-key"
$env:KEYVAULT_URL = "https://your-vault.vault.azure.net/"
$env:API_KEY_SECRET_NAME = "api-key-secret"

# Linux/macOS
export CHAT_SERVER_HOST="0.0.0.0"
export CHAT_SERVER_API_KEY="your-secure-api-key"
export KEYVAULT_URL="https://your-vault.vault.azure.net/"
export API_KEY_SECRET_NAME="api-key-secret"
```

### Customizing Model Parameters

```yaml
# Custom model_config.yaml for different use cases
model:
  id: "microsoft/DialoGPT-large"  # Different model
  device: "cuda:0"                # Specific GPU
  parameters:
    max_new_tokens: 2048          # Shorter responses
    temperature: 0.3              # More deterministic
    top_p: 0.8                    # Focused sampling
    do_sample: True
    repetition_penalty: 1.1       # Reduce repetition
```

### Adding Custom System Prompts

```yaml
# Extended system_prompt_config.yaml
system_prompts:
  assistant: |
    # Existing assistant prompt...
    
  code_reviewer: |
    # Code Review Assistant
    You are a senior software engineer specializing in code review.
    Focus on code quality, best practices, and security considerations.
    
  technical_writer: |
    # Technical Documentation Assistant  
    You are a technical writer who creates clear, comprehensive documentation.
    Focus on accuracy, clarity, and practical examples.
```
"""