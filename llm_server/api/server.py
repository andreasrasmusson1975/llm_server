"""
FastAPI server for LLM-based conversation.

Features:
- Loads Hugging Face model/tokenizer once into GPU memory.
- Stateless: client provides prompt + history; API returns reply.
- Two endpoints: blocking and streaming.
- API key authentication via Azure Key Vault secret.
- Designed for deployment behind Nginx with TLS on an Azure GPU VM.

Author: Andreas Rasmusson
"""

import os
import torch
from fastapi import FastAPI, Request, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM

# Azure Key Vault
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from conversation_engine.conversation_engine import ConversationEngine
from helper_functionality.yaml_loading import (
    load_system_prompt, 
    load_chat_server_config,
    load_model_config,
    load_azure_config
)
from helper_functionality.model_loading import initialize
import uvicorn

# ----------------------------
# API Key from Azure Key Vault
# ----------------------------
def load_api_key() -> str:
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=KV_URI, credential=credential)
        api_key = client.get_secret(SECRET_NAME).value
        print("✅ API key loaded from Azure Key Vault.")
        return api_key
    except Exception as e:
        raise RuntimeError(f"Failed to load API key from Key Vault: {e}")

def check_api_key(x_api_key: str = Header(...)):
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

KEY_VAULT_NAME = os.getenv(azure_config.get("key_vault_name"))
SECRET_NAME = os.getenv(azure_config.get("secret_name"))
KV_URI = f"https://{KEY_VAULT_NAME}.vault.azure.net/"

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
app = FastAPI(title="Conversation API", version="1.0")


@app.post("/chat/blocking")
async def chat_blocking(request: Request, auth=Depends(check_api_key)):
    """
    Blocking endpoint: waits until full reply is generated.
    """
    data = await request.json()
    user_message = data["prompt"]
    history = data.get("history", [])
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
    Streaming endpoint: yields reply chunks as they are generated.
    """
    data = await request.json()
    user_message = data["prompt"]
    history = data.get("history", [])
    improvement = data.get("improvement", False)
    intermediate_steps = data.get("intermediate_steps", False)

    engine = ConversationEngine(tok, model, system_prompt,
                                improvement=improvement,
                                intermediate_steps=intermediate_steps)
    engine.conversation_history = history

    def generate():
        for chunk in engine.stream_reply(user_message):
            yield chunk
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
    Blocking GET endpoint: quick browser-friendly way to test.
    Example:
    https://myapi.example.com/chat/blocking?prompt=Hello&x_api_key=YOUR_KEY
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
    Health check endpoint.
    Example:
    https://myapi.example.com/ping?x_api_key=YOUR_KEY
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"status": "ok"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000,reload=False)

if __name__ == "__main__":
    main()
