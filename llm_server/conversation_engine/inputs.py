"""
inputs.py
---------
Utilities for preparing and formatting inputs for conversational LLM pipelines.

This module provides helper functions to construct, format, and tokenize prompts for large language models (LLMs) in chat-based applications. It supports building model-ready inputs from system prompts, user messages, and conversation history, as well as generating review prompts for answer refinement workflows.

Key Functions:
    - prepare_inputs: Assemble and tokenize chat prompts for LLM generation.
    - build_review_message: Create structured review prompts for iterative answer improvement.

Intended for use with Hugging Face Transformers and custom conversational pipelines that require flexible prompt engineering and multi-stage review.

Author: Andreas Rasmusson
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any

def prepare_inputs(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt_dict: dict,
    message: str,
    role: str = "assistant",
    conversation_history: list | None = None
) -> dict:
    """
    Prepare model inputs for chat-style generation with a conversational LLM.

    This function constructs a prompt from the system prompt, conversation history, and user message,
    then tokenizes it for input to a Hugging Face language model. The output is ready to be passed
    directly to the model's generate method.

    Args:
        tok (Any):
            The tokenizer instance used for encoding and decoding text (e.g., Hugging Face AutoTokenizer).
        model (Any):
            The language model instance (e.g., Hugging Face AutoModelForCausalLM).
        system_prompt_dict (dict):
            Dictionary containing system prompt configuration and metadata.
        message (str):
            The user message or prompt to which the assistant should reply.
        role (str, optional):
            The role for the reply (e.g., "assistant"). Defaults to "assistant".
        conversation_history (list | None, optional):
            List of previous conversation turns, if any, to provide context. Defaults to None.

    Returns:
        dict: Tokenized model inputs, ready for use with the model's generate method.

    Example:
        >>> inputs = prepare_inputs(tok, model, system_prompt_dict, "How do I use this API?")
        >>> model.generate(**inputs)
    """
    conversation_history = conversation_history or []
    messages = [{"role": "system", "content": system_prompt_dict[role]}]
    messages += conversation_history
    messages.append({"role": "user", "content": message})
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(prompt, return_tensors="pt").to(model.device)

def build_review_message(user_message: str, draft_answer: str) -> str:
    """
    Construct a review prompt for an LLM to critique and improve a draft answer to a user message.

    This function generates a detailed prompt that asks the LLM to:
      - Review the provided draft answer in the context of the original user message.
      - Suggest at least five specific improvements in a numbered list.
      - Revise the draft answer by incorporating those improvements, while maintaining the original style and not making the answer shorter or more concise.
      - Return the output in a strict format with three sections: '### Improvements', '### Revised Answer', and '### Comments'.

    Args:
        user_message (str): The original message or question from the user.
        draft_answer (str): The initial draft answer generated for the user message.

    Returns:
        str: A formatted prompt string instructing the LLM to review and improve the draft answer, including explicit output formatting requirements.

    Example:
        >>> prompt = build_review_message("What is the capital of France?", "The capital of France is Paris.")
        >>> print(prompt)
    """
    
    return f"""
    I received the following message/question:
    {user_message}

    I thought about giving the following answer:

    {draft_answer}

    But I'm not sure if my answer fully addresses the question or if there are better ways to respond.
    Could you please help me review my answer and suggest improvements? I don't want the answer to be
    shorter or more concise. I'm more worried about not having captured the full complexity of the question.

    First give a numbered list of at least five improvements. Then incorporate those improvements 
    into the existing draft answer. Keep the style of the draft answer. 

    You MUST use the following output format:

    ### Improvements
    1. Improvement 1
    2. Improvement 2
    3. Improvement 3
    4. Improvement 4
    5. Improvement 5

    ### Revised Answer
    Revised answer here.

    ### Comments
    Comments from you on the changes.

    Your answer MUST consist of these three headings and their content below them.
    *** Very important ***
    1. The content under the heading Revised Answer MUST only contain the contents of the revised answer so, that I
       can copy-paste it.
    2. ALWAYS give detailed explanations after code snippets
    """
