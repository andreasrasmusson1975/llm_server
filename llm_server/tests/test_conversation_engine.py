import pytest
from llm_server.conversation_engine.conversation_engine import ConversationEngine
from llm_server.helper_functionality.model_loading import initialize

@pytest.fixture(scope="session")
def engine():
    from llm_server.helper_functionality.model_loading import initialize
    tok, model = initialize(model_id="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda")  # adjust to your env

    system_prompt = {
        "system": "You are a helpful assistant.",
        "user": "{message}",
        "assistant": "",
        "reviser": "You are a careful editor. Improve clarity, correctness, and style.",
        "commenter": "Provide constructive comments on the draft answer."
    }
    return ConversationEngine(tok, model, system_prompt)



def test_basic_stream_reply(engine):
    chunks = list(engine.stream_reply("Hello, are you alive?"))
    reply = "".join(chunks)

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert engine.conversation_history[-2]["role"] == "user"
    assert engine.conversation_history[-1]["role"] == "assistant"


def test_improvement_stream_reply(engine):
    engine.set_improvement(True)
    chunks = list(engine.stream_reply("Please improve this answer."))
    reply = "".join(chunks)

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert "revision2" in engine.last_result
    assert engine.conversation_history[-1]["role"] == "assistant"


def test_intermediate_steps_stream_reply(engine):
    engine.set_improvement(True).set_intermediate_steps(True)
    chunks = list(engine.stream_reply("Show me the steps."))
    reply = "".join(chunks)

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert "revision2" in engine.last_result
    assert engine.conversation_history[-1]["role"] == "assistant"
