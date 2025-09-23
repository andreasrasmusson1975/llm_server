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

The server loads a model once at startup and maintains it in memory for efficient request processing. Configuration is externalized to YAML files

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
print(f"Final reply: {engine.last_result.get('revision2', 'No revision')}")
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
    model_id="meta-llama/Llama-3.1-8B-Instruct"
)
```

### Health Check and Testing

```bash
# Health check
curl "http://localhost:8000/ping?x_api_key=your-key"

# Quick test with GET endpoint
curl "http://localhost:8000/chat/blocking?prompt=Hello&x_api_key=your-key"
```

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

# Conversation Engine Package

## Introduction

The `conversation_engine` package provides a conversational framework for Large Language Models (LLMs) built on Hugging Face Transformers. It manages multi-turn conversations, supports streaming and blocking text generation, and includes optional multi-pass answer improvement workflows. The package handles conversation history, system prompts, and provides both simple single-response generation and multi-stage refinement pipelines.

The engine is designed for applications requiring natural conversation flow with context awareness, such as chatbots, virtual assistants, and interactive AI systems.

## Core Components

### ConversationEngine (`conversation_engine.py`)
The main orchestrator class that manages conversation state and coordinates the generation pipeline:
- **Conversation History Management**: Maintains context across multiple turns
- **Streaming Control**: Handles both streaming and blocking response generation
- **Pipeline Configuration**: Controls whether to use simple generation or multi-pass improvement

### Output Generation (`outputs.py`)
Functions for text generation with different complexity levels:
- **Basic Generation**: Simple blocking text generation with `generate_reply()`
- **Streaming Generation**: Real-time token streaming with `stream_generate()`
- **Single-Pass Streaming**: Context-aware streaming with `stream_reply()`
- **Multi-Pass Improvement**: Two-stage refinement with `stream_and_improve_reply()`

### Input Processing (`inputs.py`)
Utilities for preparing model inputs and constructing prompts:
- **Chat Template Application**: Converts conversations to model-ready prompts
- **Context Assembly**: Combines system prompts, history, and user messages
- **Review Prompt Construction**: Builds structured prompts for answer improvement
- **Device Management**: Handles tensor placement for GPU/CPU execution

## Features

### 🔄 Conversation Management
- **Multi-Turn Context**: Maintains conversation history for context-aware responses
- **System Prompt Integration**: Supports configurable system prompts and roles
- **History Tracking**: Automatic conversation state management and updates

### ⚡ Generation Modes
- **Streaming Output**: Real-time token generation with chunk-by-chunk delivery
- **Blocking Mode**: Traditional request-response for simpler integrations
- **Configurable Parameters**: Temperature, top-p, max tokens, and other generation settings

### 🔧 Answer Improvement Pipeline
- **Two-Pass Refinement**: Draft generation followed by structured review and revision
- **Intermediate Transparency**: Optional display of review steps and suggestions
- **Structured Output**: Consistent formatting for improvements, revisions, and comments
- **Quality Enhancement**: Systematic approach to improving response quality and coverage

### 🎯 Flexible Configuration
- **Runtime Adjustment**: Toggle improvement and intermediate steps during operation
- **Device Flexibility**: Support for CPU and GPU execution
- **Model Agnostic**: Works with any Hugging Face causal language model

## Technical Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ User Message    │───▶│ ConversationEngine│───▶│ Input Processing│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Generation Mode  │◄───│ Prepared Inputs │
                       │ Selection        │    └─────────────────┘
                       └──────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
          ┌─────────────┐ ┌──────────┐ ┌─────────────┐
          │ Basic Reply │ │ Improved │ │ Transparent │
          │ Generation  │ │ Reply    │ │ Improvement │
          └─────────────┘ └──────────┘ └─────────────┘
                    │           │           │
                    └───────────┼───────────┘
                                ▼
                       ┌──────────────────┐
                       │ Streaming Output │
                       └──────────────────┘
```

### Generation Pipeline Flow
1. **Input Preparation**: User message combined with system prompt and conversation history
2. **Mode Selection**: Choose between basic generation or multi-pass improvement
3. **Text Generation**: Model generates response using specified parameters
4. **Optional Improvement**: Two-stage review and refinement process
5. **Output Streaming**: Real-time delivery of generated tokens
6. **State Update**: Conversation history updated with new turn

### Improvement Workflow
1. **Draft Generation**: Initial response to user message
2. **First Review**: Structured analysis with numbered improvement suggestions
3. **First Revision**: Improved response incorporating suggestions
4. **Second Review**: Additional refinement pass
5. **Final Revision**: Polished final response

## Usage

### Basic Setup

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_server.conversation_engine.conversation_engine import ConversationEngine

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",device_map="auto")

# System prompt configuration
system_prompt = {
    "assistant": "You are a helpful AI assistant. Provide clear, accurate responses.",
    "reviser": "You revise draft answers. Provide clear, accurate responses."
}

# Create conversation engine
engine = ConversationEngine(tokenizer, model, system_prompt)
```

### Simple Conversation

```python
# Basic streaming response
for chunk in engine.stream_reply("Hello, how are you today?"):
    print(chunk, end="", flush=True)

print("Final result:", engine.last_result["reply"])
```

### Multi-Turn Conversation

```python
# Set initial conversation history
engine.conversation_history = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Nice to meet you, Alice! How can I help you today?"}
]

# Continue conversation with context
for chunk in engine.stream_reply("What's my name?"):
    print(chunk, end="", flush=True)

# History is automatically updated
print(f"Conversation turns: {len(engine.conversation_history)}")
```

### Answer Improvement Mode

```python
# Enable multi-pass improvement
engine.set_improvement(True)

for chunk in engine.stream_reply("Explain quantum computing"):
    print(chunk, end="", flush=True)

# Access improvement details
result = engine.last_result
print("Draft:", result["draft_answer"])
print("Suggestions:", result["suggestions1"])
print("Final:", result["revision2"])
```

### Transparent Improvement Process

```python
# Show intermediate steps during improvement
engine.set_improvement(True).set_intermediate_steps(True)

for chunk in engine.stream_reply("How does machine learning work?"):
    print(chunk, end="", flush=True)
```

### Custom Input Preparation

```python
from llm_server.conversation_engine.inputs import prepare_inputs
import torch

# Prepare inputs manually
conversation_history = [
    {"role": "user", "content": "Explain recursion"},
    {"role": "assistant", "content": "Recursion is when a function calls itself..."}
]

inputs = prepare_inputs(
    tokenizer, 
    model, 
    system_prompt, 
    "Can you give an example?",
    conversation_history=conversation_history
)

# Use with model directly
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=512)
    print(output)
```

### Streaming with Custom Parameters

```python
from llm_server.conversation_engine.outputs import stream_generate

# Custom generation parameters
inputs = prepare_inputs(tokenizer, model, system_prompt, "Write a poem")

for chunk in stream_generate(inputs, tokenizer, model, max_new_tokens=1024):
    if chunk is not None:
        print(chunk, end="", flush=True)
```

### Error Handling

```python
try:
    for chunk in engine.stream_reply("Complex question about AI"):
        print(chunk, end="", flush=True)
except Exception as e:
    print(f"Generation failed: {e}")
    # Conversation history remains unchanged on failure
```

### Batch Processing

```python
questions = [
    "What is artificial intelligence?",
    "How do neural networks work?",
    "Explain deep learning"
]

responses = []
for question in questions:
    for chunk in engine.stream_reply(question):
        pass  # Process streaming chunks as needed
    
    responses.append(engine.last_result["reply"])
    print(f"Completed question {len(responses)}/{len(questions)}")
```

### Performance Optimization

```python
import torch

# Use inference mode for better performance
with torch.inference_mode():
    for chunk in engine.stream_reply("Optimization question"):
        print(chunk, end="", flush=True)

# Clear CUDA cache periodically for long-running applications
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

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

### Customizing Model Parameters

```yaml
# Custom model_config.yaml for different use cases
model:
  id: "meta-llama/Meta-Llama-3-8B"  # Different model
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
