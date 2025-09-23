"""
# API Package

## Introduction

The `api` package provides a REST API server for conversational AI interactions using large language models. Built on FastAPI, it exposes endpoints for both blocking and streaming conversations with configurable answer improvement and intermediate step visibility.

## Core Components

- **FastAPI Server**: HTTP server exposing conversation endpoints
- **Authentication**: Azure Key Vault integration for secure API key management
- **Model Integration**: Direct integration with Hugging Face transformers for text generation
- **Conversation Engine**: High-level interface for managing LLM interactions and conversation history

## Features

- **Multiple Interaction Modes**: Blocking and streaming response endpoints
- **Conversation History**: Maintains context across multiple turns
- **Answer Improvement**: Optional multi-pass refinement of generated responses
- **Intermediate Steps**: Configurable visibility into the reasoning process
- **Secure Authentication**: API key validation using Azure Key Vault
- **Browser-Friendly Testing**: GET endpoints for quick testing and debugging

## Technical Architecture

The API layer acts as the HTTP interface to the conversation engine, handling:

- Request validation and authentication
- JSON payload processing
- Response streaming and formatting
- Error handling and status codes
- Model lifecycle management (load once, serve many)

The server loads the language model and tokenizer at startup, maintaining them in memory for efficient request processing. Configuration is managed through YAML files covering model parameters, server settings, and system prompts.

## Usage

### Starting the Server

```python
from llm_server.api.server import main

# Start the server on default host (0.0.0.0) and port (8000)
main()
```

### Blocking Conversation (POST)

```python
import requests

url = "http://localhost:8000/chat/blocking"
headers = {"X-API-Key": "your-api-key"}
payload = {
    "prompt": "Explain quantum computing",
    "history": [],
    "improvement": False,
    "intermediate_steps": False
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(result["reply"])
```

### Streaming Conversation (POST)

```python
import requests

url = "http://localhost:8000/chat/stream"
headers = {"X-API-Key": "your-api-key"}
payload = {
    "prompt": "Write a short story",
    "history": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"}
    ],
    "improvement": True,
    "intermediate_steps": True
}

response = requests.post(url, json=payload, headers=headers, stream=True)
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    if chunk and chunk != "[[END]]":
        print(chunk, end="", flush=True)
```

### Quick Testing (GET)

```bash
# Browser-friendly endpoint for quick testing
curl "http://localhost:8000/chat/blocking?prompt=Hello&x_api_key=your-key"

# Health check
curl "http://localhost:8000/ping?x_api_key=your-key"
```

### With Answer Improvement

```python
payload = {
    "prompt": "Explain machine learning",
    "improvement": True,        # Enable multi-pass improvement
    "intermediate_steps": True  # Show reasoning steps
}
```
"""