"""
# LLM Server

## Introduction

LLM Server is a Python package for deploying and managing large language models through a REST API. It provides a FastAPI-based server that handles conversational AI interactions with support for streaming responses, conversation history, and multi-pass answer improvement. The package integrates Hugging Face transformers with performance optimizations and secure credential management.

## Core Components

- **API Server**: FastAPI-based REST API with blocking and streaming endpoints
- **Conversation Engine**: High-level interface for managing LLM interactions and conversation flow
- **Model Loading**: Optimized initialization of Hugging Face models with performance enhancements
- **Configuration Management**: YAML-based configuration system for models, server settings, and prompts
- **Azure Integration**: Secure API key management through Azure Key Vault
- **Input/Output Processing**: Structured handling of conversation context and response formatting

## Features

- **Multiple Response Modes**: Both blocking and streaming response endpoints
- **Conversation Context**: Maintains conversation history for context-aware responses
- **Answer Improvement**: Optional multi-pass refinement with intermediate step visibility
- **Performance Optimization**: Model compilation, Flash Attention 2, and optional quantization
- **Secure Authentication**: API key validation with Azure Key Vault integration
- **Configurable Prompts**: Role-based system prompts for different assistant behaviors
- **Browser-Friendly Testing**: GET endpoints for quick API testing
- **Health Monitoring**: Built-in health check endpoint

## Technical Architecture

The LLM Server follows a modular architecture with clear separation of concerns:

**API Layer** (`api/`): FastAPI server handling HTTP requests, authentication, and response formatting. Supports both JSON and streaming responses.

**Conversation Engine** (`conversation_engine/`): Core business logic for managing LLM interactions, including conversation history, streaming generation, and optional answer improvement pipelines.

**Helper Functionality** (`helper_functionality/`): Utility modules for model initialization with performance optimizations and YAML configuration loading with structured access patterns.

**Configuration** (`yaml_files/`): Centralized YAML-based configuration for model parameters, server settings, Azure integration, and system prompts.

The server loads models once at startup and maintains them in memory for efficient request processing. Configuration is externalized to YAML files, supporting environment variable substitution for deployment flexibility.

## Usage

### Starting the Server

```python
from llm_server.api.server import main

# Start server with default configuration
main()
```

### Command Line Interface

```bash
# Start server using console script
start_llm_server

# Server will start on 0.0.0.0:8000 by default
```

### Basic API Usage

```python
import requests

# Blocking conversation
url = "http://localhost:8000/chat/blocking"
headers = {"X-API-Key": "your-api-key"}
payload = {
    "prompt": "Explain how REST APIs work",
    "history": [],
    "improvement": False,
    "intermediate_steps": False
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(result["reply"])
```

### Streaming Responses

```python
import requests

url = "http://localhost:8000/chat/stream"
headers = {"X-API-Key": "your-api-key"}
payload = {
    "prompt": "Write a Python function to calculate fibonacci numbers",
    "improvement": True,
    "intermediate_steps": True
}

response = requests.post(url, json=payload, headers=headers, stream=True)
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    if chunk and chunk != "[[END]]":
        print(chunk, end="", flush=True)
```

### Using the Conversation Engine Directly

```python
from llm_server.conversation_engine.conversation_engine import ConversationEngine
from llm_server.helper_functionality.model_loading import initialize
from llm_server.helper_functionality.yaml_loading import (
    load_model_config, 
    load_system_prompt
)

# Load configuration and model
config = load_model_config()
system_prompt = load_system_prompt()
tokenizer, model = initialize(
    device=config.get("device", "auto"),
    model_id=config.get("id")
)

# Create conversation engine
engine = ConversationEngine(
    tok=tokenizer,
    model=model,
    system_prompt=system_prompt,
    improvement=True,
    intermediate_steps=True
)

# Generate streaming response
for chunk in engine.stream_reply("How do neural networks learn?"):
    print(chunk, end="", flush=True)

# Access final result
print(f"\nFinal reply: {engine.last_result.get('revision2', 'No revision')}")
```

### Conversation with History

```python
# Maintain conversation context
conversation_history = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."}
]

engine.conversation_history = conversation_history
for chunk in engine.stream_reply("Can you give me a practical example?"):
    print(chunk, end="", flush=True)
```

### Configuration Customization

```python
from llm_server.helper_functionality.yaml_loading import load_yaml

# Load custom configuration
custom_config = load_yaml("custom_model_config.yaml")

# Initialize with custom settings
tokenizer, model = initialize(
    device="cuda:0",
    model_id="microsoft/DialoGPT-large"
)
```

### Health Check and Testing

```bash
# Health check
curl "http://localhost:8000/ping?x_api_key=your-key"

# Quick test with GET endpoint
curl "http://localhost:8000/chat/blocking?prompt=Hello&x_api_key=your-key"
```

### Environment Configuration

```bash
# Set environment variables for deployment
export CHAT_SERVER_HOST="0.0.0.0"
export CHAT_SERVER_API_KEY="your-secure-api-key"
export KEYVAULT_URL="https://your-vault.vault.azure.net/"
export API_KEY_SECRET_NAME="api-key-secret"
```
"""