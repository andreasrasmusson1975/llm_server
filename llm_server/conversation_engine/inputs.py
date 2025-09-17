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
    
    Please review my answer and suggest improvements. I do NOT want a shorter answer; I want full coverage of the user’s question and concerns.
    
    First, give a numbered list of at least five improvements. Then produce a full, self-contained revised answer that a user could send as-is (not a diff). Keep the original structure where possible (intro → code → explanation → outro), and keep the original voice/tone.
    
    You MUST use the following output format:
    
    ### Improvements
    1. Improvement 1
    2. Improvement 2
    3. Improvement 3
    4. Improvement 4
    5. Improvement 5
    
    ### Revised Answer
    (Output the ENTIRE revised message here — including any intro text, code blocks, explanations immediately after each code block, and any closing remarks. Do NOT output only changed lines, a diff, or a snippet. No ellipses. Preserve any prose that was in the draft unless removing redundancy; if you remove something, justify it in Comments. Include at least one paragraph BEFORE the first code block and at least one AFTER the last code block, unless the draft lacked them. Keep code fences with language tags, e.g., ```python. The revised answer must not be shorter overall than the draft. Explanations intended for the user belong here, not in Comments.)
    
    ### Comments
    (Explain what you changed and why — meta commentary only. Do NOT move explanations meant for the user here.)
    
    Hard constraints:
    - In “Revised Answer”, print the full message (intro + code (if any) + explanations + closing), not just changed lines.
    - Keep the style and tone of the draft; expand coverage and clarity without compressing content.
    - Do not omit surrounding prose if the draft had it; mirror the draft’s structure unless an improvement requires reorganization (justify in Comments).
    """
