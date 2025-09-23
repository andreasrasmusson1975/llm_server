
"""
LLM Server API Module

This module implements a FastAPI-based REST API server for serving Large Language Model
(LLM) conversations with Azure Key Vault integration for secure API key management.

The server provides both blocking and streaming endpoints for chat interactions,
supporting conversation history, response improvement, and intermediate step visualization.

Key Features:
    - FastAPI-based REST API with automatic OpenAPI documentation
    - Azure Key Vault integration for secure API key storage and retrieval
    - Support for both blocking and streaming chat responses
    - Conversation history management across multiple turns
    - Optional response improvement and intermediate step logging
    - GPU-accelerated inference using PyTorch and Transformers
    - Authentication via X-API-Key header

API Endpoints:
    POST /chat/blocking   - Synchronous chat completion
    POST /chat/stream     - Streaming chat completion
    GET  /chat/blocking   - Simple GET-based chat (for testing)
    GET  /ping           - Health check endpoint

Authentication:
    All endpoints require a valid API key passed via the 'X-API-Key' header.
    The API key is securely retrieved from Azure Key Vault on server startup.

Configuration:
    The server loads configuration from YAML files for:
    - Model settings (model ID, device)
    - Azure Key Vault settings
    - Server configuration
    - System prompts

Environment Variables:
    KEYVAULT_URL         - Azure Key Vault URL
    API_KEY_SECRET_NAME  - Name of the secret containing the API key

Author: Andreas Rasmusson
"""

import os
import torch
from fastapi import FastAPI, Request, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from llm_server.conversation_engine.conversation_engine import ConversationEngine
from llm_server.helper_functionality.yaml_loading import (
    load_system_prompt, 
    load_chat_server_config,
    load_model_config,
    load_azure_config
)
from llm_server.helper_functionality.model_loading import initialize
import uvicorn

# ----------------------------
# API Key from Azure Key Vault
# ----------------------------
def load_api_key() -> str:
    """
    Load API key from Azure Key Vault using DefaultAzureCredential.
    
    This function authenticates to Azure Key Vault using the DefaultAzureCredential
    chain, which tries multiple authentication methods in order:
    1. Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
    2. Managed Identity (when running on Azure services)
    3. Azure CLI credentials
    4. Visual Studio Code credentials
    5. Azure PowerShell credentials
    
    The function retrieves the API key secret from the configured Key Vault
    and returns it for use in API authentication.
    
    Returns:
        str: The API key retrieved from Azure Key Vault
        
    Raises:
        RuntimeError: If authentication fails or the secret cannot be retrieved
    Note:
        This function is called once during server startup. The retrieved API key
        is cached in the global API_KEY variable for subsequent authentication checks.
    """
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=KV_URI, credential=credential)
        api_key = client.get_secret(SECRET_NAME).value
        print("✅ API key loaded from Azure Key Vault.")
        return api_key
    except Exception as e:
        raise RuntimeError(f"Failed to load API key from Key Vault: {e}")

def check_api_key(x_api_key: str = Header(...)):
    """
    FastAPI dependency for validating API key authentication.
    
    This function serves as a FastAPI dependency that validates incoming requests
    by checking the X-API-Key header against the configured API key. It's used
    with FastAPI's Depends() to automatically authenticate requests to protected
    endpoints.
    
    Args:
        x_api_key (str): The API key from the X-API-Key header. The Header(...)
                        indicates this is a required header parameter that FastAPI
                        will automatically extract from the request.
    
    Returns:
        None: This function doesn't return a value. If authentication succeeds,
              the function completes normally. If it fails, an HTTPException is raised.
    
    Raises:
        HTTPException: Raised with status code 403 (Forbidden) if the provided
                      API key doesn't match the expected API key loaded from
                      Azure Key Vault.
    
    Usage:
        This function is used as a FastAPI dependency:
        
        @app.post("/protected-endpoint")
        async def protected_route(auth=Depends(check_api_key)):
            # This code only runs if authentication succeeds
            return {"message": "Access granted"}
    
    Example Request:
        curl -X POST "http://localhost:8000/chat/blocking" \
             -H "X-API-Key: your-secret-api-key" \
             -H "Content-Type: application/json" \
             -d '{"prompt": "Hello"}'
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


# ----------------------------
# Config
# ----------------------------
model_config = load_model_config()
azure_config = load_azure_config()
server_config = load_chat_server_config()
system_prompt = load_system_prompt()

MODEL_ID = model_config.get("id")
DEVICE = model_config.get("device")

# Env vars for Key Vault
KV_URI = os.getenv("KEYVAULT_URL")
SECRET_NAME = os.getenv("API_KEY_SECRET_NAME")

API_KEY = load_api_key()

# ----------------------------
# Load model/tokenizer once
# ----------------------------
print(f"🚀 Loading model {MODEL_ID} onto {DEVICE}...")
tok, model = initialize(model_id=MODEL_ID, device=DEVICE)   
print("✅ Model loaded and ready.")


# ----------------------------
# FastAPI app
# ----------------------------
print("""🚀 Starting FastAPI server...""")
app = FastAPI(title="Conversation API", version="1.0")
print("✅ FastAPI server started.")

@app.post("/chat/blocking")
async def chat_blocking(request: Request, auth=Depends(check_api_key)):
    """
    Synchronous chat completion endpoint with blocking response.
    
    This endpoint processes a chat message and returns the complete response
    after the LLM has finished generating. Unlike the streaming endpoint,
    this returns the full response in a single JSON object once generation
    is complete.
    
    The endpoint supports conversation history, response improvement, and
    intermediate step logging for enhanced conversation quality and debugging.
    
    Args:
        request (Request): FastAPI request object containing the JSON payload
        auth: Authentication dependency that validates the X-API-Key header
    
    Request Body (JSON):
        prompt (str): The user's message/question to the LLM
        history (list, optional): Previous conversation turns as a list of
                                 message objects. Defaults to empty list.
        improvement (bool, optional): Enable response improvement/revision.
                                    Defaults to False.
        intermediate_steps (bool, optional): Enable logging of intermediate
                                           processing steps. Defaults to False.
    
    Returns:
        JSONResponse: A JSON object containing:
            - reply (str): The LLM's complete response to the user's prompt
    
    Response Format:
        {
            "reply": "The LLM's complete response text"
        }
    
    Raises:
        HTTPException: 403 Forbidden if authentication fails
        HTTPException: 422 Unprocessable Entity if request body is invalid
    
    Example Request:
        POST /chat/blocking
        Headers:
            X-API-Key: your-secret-api-key
            Content-Type: application/json
        Body:
            {
                "prompt": "What is machine learning?",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ],
                "improvement": true,
                "intermediate_steps": false
            }
    
    Example Response:
        {
            "reply": "Machine learning is a subset of artificial intelligence..."
        }
    
    Note:
        This endpoint waits for the complete response generation before returning,
        which may take several seconds depending on the prompt complexity and
        model size. For real-time streaming responses, use /chat/stream instead.
    """
    data = await request.json()
    user_message = data["prompt"]
    history = data.get("history") or []
    improvement = data.get("improvement", False)
    intermediate_steps = data.get("intermediate_steps", False)

    engine = ConversationEngine(tok, model, system_prompt,
                                improvement=improvement,
                                intermediate_steps=intermediate_steps)
    engine.conversation_history = history

    chunks = []
    for chunk in engine.stream_reply(user_message):
        chunks.append(chunk)

    result = engine.last_result
    reply = result.get("revision2") or result.get("reply")

    return JSONResponse({"reply": reply})


@app.post("/chat/stream")
async def chat_stream(request: Request, auth=Depends(check_api_key)):
    """
    Streaming chat completion endpoint with real-time response generation.
    
    This endpoint processes a chat message and streams the response in real-time
    as the LLM generates tokens. This provides a better user experience for
    longer responses by showing partial results immediately instead of waiting
    for the complete response.
    
    The endpoint supports the same features as the blocking version: conversation
    history, response improvement, and intermediate step logging.
    
    Args:
        request (Request): FastAPI request object containing the JSON payload
        auth: Authentication dependency that validates the X-API-Key header
    
    Request Body (JSON):
        prompt (str): The user's message/question to the LLM
        history (list, optional): Previous conversation turns as a list of
                                 message objects. Defaults to empty list.
        improvement (bool, optional): Enable response improvement/revision.
                                    Defaults to False.
        intermediate_steps (bool, optional): Enable logging of intermediate
                                           processing steps. Defaults to False.
    
    Returns:
        StreamingResponse: A streaming text response where each chunk contains
                          partial text as it's generated, ending with "[[END]]"
                          marker to indicate completion.
    
    Response Format:
        - Content-Type: text/plain
        - Body: Stream of text chunks followed by "[[END]]" terminator
        - Each chunk represents partial generated text
        - Final chunk is always "[[END]]" to signal completion
    
    Raises:
        HTTPException: 403 Forbidden if authentication fails
        HTTPException: 422 Unprocessable Entity if request body is invalid
    
    Example Request:
        POST /chat/stream
        Headers:
            X-API-Key: your-secret-api-key
            Content-Type: application/json
        Body:
            {
                "prompt": "Tell me about quantum computing",
                "history": [],
                "improvement": false,
                "intermediate_steps": false
            }
    
    Example Response Stream:
        "Quantum"
        " computing"
        " is a revolutionary"
        " technology that..."
        "[[END]]"
    
    Usage Notes:
        - Clients should read the stream until they receive "[[END]]"
        - Each chunk should be appended to build the complete response
        - The stream may include intermediate processing output if enabled
        - Connection should be kept alive during streaming
        - Consider implementing timeout handling for long responses
    
    Client Example (JavaScript):
        const response = await fetch('/chat/stream', {
            method: 'POST',
            headers: {
                'X-API-Key': 'your-key',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({prompt: 'Hello'})
        });
        
        const reader = response.body.getReader();
        let result = '';
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            const chunk = new TextDecoder().decode(value);
            if (chunk === '[[END]]') break;
            result += chunk;
        }
    """
    data = await request.json()
    user_message = data["prompt"]
    history = data.get("history") or []
    improvement = data.get("improvement", False)
    intermediate_steps = data.get("intermediate_steps", False)

    engine = ConversationEngine(tok, model, system_prompt,
                                improvement=improvement,
                                intermediate_steps=intermediate_steps)
    engine.conversation_history = history

    def generate():
        for chunk in engine.stream_reply(user_message):
            yield chunk
        result = engine.last_result
        final_reply = result.get("revision2") or result.get("reply")
        yield "[[END]]"

    return StreamingResponse(generate(), media_type="text/plain")
@app.get("/chat/blocking")
async def chat_blocking_get(
    prompt: str,
    x_api_key: str,
    improvement: bool = False,
    intermediate_steps: bool = False
):
    """
    Simple GET-based chat completion endpoint for testing and basic integrations.
    
    This endpoint provides a simplified interface for chat completion using URL
    query parameters instead of a JSON request body. It's designed for easy
    testing, debugging, and integration with systems that prefer GET requests
    over POST requests with JSON payloads.
    
    Unlike the POST version, this endpoint doesn't support conversation history
    and always starts with a fresh conversation context.
    
    Args:
        prompt (str): The user's message/question to the LLM (required query parameter)
        x_api_key (str): API key for authentication (required query parameter)
        improvement (bool, optional): Enable response improvement/revision.
                                    Defaults to False.
        intermediate_steps (bool, optional): Enable logging of intermediate
                                           processing steps. Defaults to False.
    
    Returns:
        dict: A JSON object containing:
            - reply (str): The LLM's complete response to the user's prompt
    
    Response Format:
        {
            "reply": "The LLM's complete response text"
        }
    
    Raises:
        HTTPException: 403 Forbidden if authentication fails
        HTTPException: 422 Unprocessable Entity if required parameters are missing
    
    Example Request:
        GET /chat/blocking?prompt=Hello%20world&x_api_key=your-secret-key&improvement=true
    
    Example Response:
        {
            "reply": "Hello! How can I help you today?"
        }
    
    URL Example:
        http://localhost:8000/chat/blocking?prompt=What%20is%20AI&x_api_key=abc123
    
    Usage Notes:
        - The prompt parameter must be URL-encoded for special characters
        - No conversation history is maintained between requests
        - Each request starts a fresh conversation context
        - Primarily intended for testing, debugging, and simple integrations
        - For production applications with conversation history, use POST /chat/blocking
        - Boolean parameters accept: true/false, 1/0, yes/no (case-insensitive)
    
    Testing Examples:
        # Basic test
        curl "http://localhost:8000/chat/blocking?prompt=Hello&x_api_key=key123"
        
        # With improvement enabled
        curl "http://localhost:8000/chat/blocking?prompt=Explain%20AI&x_api_key=key123&improvement=true"
        
        # Browser test
        http://localhost:8000/chat/blocking?prompt=Test&x_api_key=your-key
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    engine = ConversationEngine(tok, model, system_prompt,
                                improvement=improvement,
                                intermediate_steps=intermediate_steps)

    engine.conversation_history = []
    chunks = []
    for chunk in engine.stream_reply(prompt):
        chunks.append(chunk)

    result = engine.last_result
    reply = result.get("revision2") or result.get("reply")

    return {"reply": reply}


@app.get("/ping")
async def ping(x_api_key: str):
    """
    Health check endpoint for server status and authentication validation.
    
    This endpoint provides a simple way to verify that the server is running,
    accessible, and that authentication is working correctly. It's commonly
    used for monitoring, load balancer health checks, and integration testing.
    
    The endpoint performs minimal processing and returns quickly, making it
    ideal for automated health monitoring systems that need to verify service
    availability without consuming significant resources.
    
    Args:
        x_api_key (str): API key for authentication (required query parameter)
    
    Returns:
        dict: A simple JSON object indicating successful response:
            - status (str): Always "ok" if the request succeeds
    
    Response Format:
        {
            "status": "ok"
        }
    
    Raises:
        HTTPException: 403 Forbidden if the provided API key is invalid
    
    Example Request:
        GET /ping?x_api_key=your-secret-api-key
    
    Example Response:
        {
            "status": "ok"
        }
    
    HTTP Status Codes:
        200: Server is healthy and authentication succeeded
        403: Authentication failed (invalid API key)
        500: Server error (should not occur for this simple endpoint)
    
    Usage Examples:
        # Command line test
        curl "http://localhost:8000/ping?x_api_key=abc123"
        
        # Python requests
        import requests
        response = requests.get("http://localhost:8000/ping", 
                               params={"x_api_key": "abc123"})
        assert response.json()["status"] == "ok"
        
        # Health check script
        if curl -s "http://localhost:8000/ping?x_api_key=$API_KEY" | grep -q "ok"; then
            echo "Server is healthy"
        else
            echo "Server is down"
        fi
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"status": "ok"}

def main():
    """
    Entry point for starting the LLM Server API application.
    
    This function configures and starts the Uvicorn ASGI server to serve the
    FastAPI application. It sets up the server to listen on all network
    interfaces (0.0.0.0) on port 8000, making it accessible both locally
    and from external networks.
    
    The server is configured for production use with auto-reload disabled
    for better performance and stability. All model loading, configuration
    parsing, and authentication setup occurs during module import before
    this function is called.
    
    Server Configuration:
        - Host: 0.0.0.0 (all interfaces)
        - Port: 8000
        - Reload: False (production mode)
        - Workers: 1 (single process)
    
    Prerequisites:
        Before calling this function, the following must be properly configured:
        - Environment variables: KEYVAULT_URL, API_KEY_SECRET_NAME
        - Azure authentication (managed identity, CLI, or service principal)
        - CUDA drivers and toolkit (for GPU acceleration)
        - Model files accessible to the transformers library
        - YAML configuration files in the expected locations
    
    Network Access:
        The server binds to 0.0.0.0:8000, making it accessible via:
        - http://localhost:8000 (local access)
        - http://SERVER_IP:8000 (network access)
        - Through reverse proxy (nginx) if configured
    
    Usage:
        # Direct execution
        python server.py
        
        # Module execution
        python -m llm_server.api.server
        
        # Programmatic usage
        from llm_server.api.server import main
        main()
    
    Production Deployment:
        For production, this server is typically run behind a reverse proxy:
        - Nginx handles SSL termination and static files
        - Systemd manages the process lifecycle
        - Environment variables configure Azure integration
        
    Example systemd service:
        [Unit]
        Description=LLM Server
        After=network.target
        
        [Service]
        User=llmserver
        WorkingDirectory=/opt/llm_server
        ExecStart=/opt/llm_server/venv/bin/python -m llm_server.api.server
        Restart=always
        EnvironmentFile=/etc/llm_server.env
        
        [Install]
        WantedBy=multi-user.target
    
    Monitoring:
        Once started, the server provides these endpoints for monitoring:
        - GET /ping - Health check and authentication test
        - Uvicorn access logs for request monitoring
        - Standard output for application logs
    
    Shutdown:
        The server can be stopped gracefully using:
        - Ctrl+C (SIGINT) for interactive sessions
        - SIGTERM for process management
        - systemctl stop for systemd services
    """
    uvicorn.run(app, host="0.0.0.0", port=8000,reload=False)

if __name__ == "__main__":
    main()
