import pytest
from llm_server.conversation_engine.conversation_engine import ConversationEngine
from llm_server.helper_functionality.model_loading import initialize

def engine():
    # Initialize tokenizer + model the same way your server does
    tok, model = initialize(model_id="default", device="cuda")  # adjust args as in your server
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}
    return ConversationEngine(tok, model, system_prompt)

def test_basic_stream_reply(engine):
    chunks = list(engine.stream_reply("Hello, are you alive?"))
    reply = "".join(chunks)

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert "Hello" in reply or "alive" in reply  # allow flexible model response
    assert engine.conversation_history[-2]["role"] == "user"
    assert engine.conversation_history[-1]["role"] == "assistant"

def test_improvement_stream_reply(engine):
    engine.set_improvement(True)
    chunks = list(engine.stream_reply("Please improve this answer."))
    reply = "".join(chunks)

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert "improve" in reply.lower() or "answer" in reply.lower()
    assert "revision2" in engine.last_result
    assert engine.conversation_history[-1]["role"] == "assistant"

def test_intermediate_steps_stream_reply(engine):
    engine.set_improvement(True).set_intermediate_steps(True)
    chunks = list(engine.stream_reply("Show me the steps."))
    reply = "".join(chunks)

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert "steps" in reply.lower() or "improve" in reply.lower()
    assert "revision2" in engine.last_result
    assert engine.conversation_history[-1]["role"] == "assistant"
