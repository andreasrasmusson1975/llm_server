"""
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
- **Method Chaining**: Supports fluent configuration with chainable setter methods

### Output Generation (`outputs.py`)
Functions for text generation with different complexity levels:
- **Basic Generation**: Simple blocking text generation with `generate_reply()`
- **Streaming Generation**: Real-time token streaming with `stream_generate()`
- **Single-Pass Streaming**: Context-aware streaming with `stream_reply()`
- **Multi-Pass Improvement**: Two-stage refinement with `stream_and_improve_reply()`
- **Transparent Improvement**: Multi-pass with intermediate step display

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
- **Method Chaining**: Fluent API for configuration changes
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
from llm_server.conversation_engine import ConversationEngine

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# System prompt configuration
system_prompt = {
    "assistant": "You are a helpful AI assistant. Provide clear, accurate responses."
}

# Create conversation engine
engine = ConversationEngine(tokenizer, model, system_prompt)
```

### Simple Conversation

```python
# Basic streaming response
for chunk in engine.stream_reply("Hello, how are you today?"):
    print(chunk, end="", flush=True)

print("\nFinal result:", engine.last_result["reply"])
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

### Method Chaining Configuration

```python
# Configure multiple options with method chaining
engine.set_improvement(True).set_intermediate_steps(False)

# Or configure individually
engine.set_improvement(False)
engine.set_intermediate_steps(True)
```

### Direct Function Usage

```python
from llm_server.conversation_engine.outputs import stream_reply, get_reply
from llm_server.conversation_engine.inputs import prepare_inputs

# Use functions directly without ConversationEngine wrapper
for chunk in stream_reply(tokenizer, model, system_prompt, "Hello world"):
    print(chunk, end="", flush=True)

# Blocking generation
reply = get_reply(tokenizer, model, system_prompt, "What is Python?")
print(reply)
```

### Custom Input Preparation

```python
from llm_server.conversation_engine.inputs import prepare_inputs

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

### Integration with FastAPI

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(message: str):
    def generate():
        for chunk in engine.stream_reply(message):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")
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
"""