
"""
Input Processing Module for Conversation Engine

This module provides essential input preparation and prompt construction utilities
for the LLM conversation engine. It handles the task of formatting various
types of inputs into the proper structure required by language models, including
conversation context preparation and specialized prompts for response improvement.

The module serves as the input layer between raw user messages and the tokenized
inputs required by transformer models, ensuring proper formatting, context inclusion,
and prompt engineering for optimal model performance.

Key Functions:
    - prepare_inputs: Converts conversations into tokenized model inputs
    - build_review_message: Creates structured prompts for response improvement

Core Responsibilities:
    1. Chat Template Application: Formats messages using model-specific templates
    2. Context Management: Incorporates conversation history and system prompts
    3. Tokenization: Converts text to model-compatible tensor inputs
    4. Device Management: Ensures inputs are on the correct device (CPU/GPU)
    5. Prompt Engineering: Constructs specialized prompts for improvement workflows

Input Flow:
    Raw Message → Context Addition → Template Application → Tokenization → Model Input

Conversation Structure:
    The module handles structured conversations with the following format:
    - System message (role: "system"): Contains behavioral instructions
    - History messages: Previous user/assistant exchanges for context
    - Current message (role: "user"): The immediate user input
    - Generation prompt: Signals the model to generate an assistant response

Chat Template Integration:
    Leverages Hugging Face tokenizer's apply_chat_template functionality to ensure
    proper formatting according to the specific model's expected conversation format.
    This handles model-specific tokens, separators, and formatting requirements.

Review Prompt Engineering:
    The build_review_message function implements a sophisticated prompt engineering
    pattern for multi-pass response improvement:
    - Structured critique format with numbered improvements
    - Explicit output formatting requirements
    - Style and tone preservation instructions
    - Comprehensive revision guidelines

Device Compatibility:
    All tokenized inputs are automatically moved to the same device as the model
    to ensure compatibility for GPU-accelerated inference.

Usage Examples:
    # Basic input preparation
    inputs = prepare_inputs(tokenizer, model, system_prompt, "Hello")
    outputs = model.generate(**inputs)
    
    # With conversation history
    history = [
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
    inputs = prepare_inputs(tokenizer, model, system_prompt, "What about 3+3?", 
                           conversation_history=history)
    
    # Review prompt for improvement
    review_prompt = build_review_message("Explain AI", "AI is machine learning.")

Dependencies:
    - transformers: AutoTokenizer, AutoModelForCausalLM
    - typing: Type hints for function signatures

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
