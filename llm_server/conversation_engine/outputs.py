
"""
Output Generation Module for Conversation Engine

This module implements the core text generation capabilities for the LLM conversation
engine, providing both basic response generation and sophisticated multi-pass improvement
workflows. It handles streaming and blocking generation modes, response parsing, and
automated quality enhancement through iterative refinement.

The module serves as the output layer of the conversation system, taking prepared inputs
and generating high-quality responses through various generation strategies ranging from
simple single-pass generation to complex multi-stage improvement pipelines.

Key Functions:
    Generation Functions:
        - generate_reply: Basic blocking text generation
        - stream_generate: Low-level streaming text generation
        - get_reply: High-level blocking generation with input preparation
        - stream_reply: High-level streaming generation with input preparation

    Improvement Functions:
        - get_and_improve_reply: Two-pass improvement (blocking)
        - stream_and_improve_reply: Two-pass improvement (streaming)
        - stream_and_improve_reply_display_intermediate: Streaming with visible steps
        - get_revised_reply: Single improvement pass
        - parse_review_sections: Extract structured improvement feedback

Core Capabilities:
    1. Streaming Generation: Real-time token-by-token output using TextIteratorStreamer
    2. Blocking Generation: Complete response generation for synchronous workflows
    3. Multi-Pass Improvement: Automated response refinement and enhancement
    4. Structured Parsing: Extraction of improvements, revisions, and comments
    5. Thread Management: Safe concurrent generation for streaming operations
    6. Device Optimization: GPU-accelerated inference with proper resource management

Generation Modes:
    Basic Mode:
        - Single-pass generation
        - No post-processing or improvement
        - Fastest response time
        - Suitable for simple queries

    Improvement Mode:
        - Two-pass generation (draft + revision)
        - Structured critique and enhancement
        - Higher quality responses
        - Longer response time

    Intermediate Display Mode:
        - Improvement mode with visible intermediate steps
        - Full transparency of the improvement process
        - Useful for debugging and understanding

Streaming Architecture:
    The streaming system uses a producer-consumer pattern with background threads:
    1. Main thread yields chunks as they're produced
    2. Background thread runs model.generate() with streamer callback
    3. TextIteratorStreamer handles token-to-text conversion
    4. Thread synchronization ensures proper cleanup

Improvement Pipeline:
    1. Generate initial draft response
    2. Create structured review prompt
    3. Generate critique with numbered improvements
    4. Parse improvements, revised answer, and comments
    5. Return structured result with all components

Response Quality Enhancement:
    The improvement system implements sophisticated prompt engineering to:
    - Maintain original style and tone
    - Expand content coverage without compression
    - Provide structured, actionable feedback
    - Generate comprehensive revisions

Configuration Parameters:
    All generation functions use optimized parameters:
    - max_new_tokens: 4096 (configurable)
    - temperature: 0.7 (balanced creativity/consistency)
    - top_p: 0.9 (nucleus sampling)
    - do_sample: True (stochastic generation)

Error Handling:
    - Graceful degradation for parsing failures
    - Thread safety for concurrent operations
    - Device compatibility checks
    - Memory management for long sequences

Dependencies:
    - transformers: AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    - torch: GPU acceleration and inference mode
    - threading: Background generation threads
    - llm_server.conversation_engine.inputs: Input preparation utilities

Author: Andreas Rasmusson
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading
from llm_server.conversation_engine.inputs import *
from typing import List, Tuple, Optional
from collections.abc import Generator


def generate_reply(
        inputs: dict, 
        tok: AutoTokenizer, 
        model: AutoModelForCausalLM
) -> str:
    """
    Generate a text reply from a causal language model given preprocessed inputs.

    This function performs a forward pass using the provided model and tokenizer
    to generate a reply based on the input prompt. It uses inference mode for 
    efficiency and disables gradient computation. The function is designed for 
    single-turn, non-streaming generation and returns the decoded text output, 
    excluding the input prompt and any special tokens.

    Args:
        inputs (dict):
            A dictionary of model inputs.
        tok (AutoTokenizer):
            The Hugging Face tokenizer instance used for encoding and decoding text.
        model (AutoModelForCausalLM):
            The Hugging Face causal language model instance used for text generation.

    Returns:
        str: The generated reply as a string, with special tokens removed and leading/trailing 
             whitespace stripped.

    Example:
        >>> inputs = tok("Hello, how are you?", return_tensors="pt")
        >>> reply = generate_reply(inputs, tok, model)
        >>> print(reply)
        "I'm doing well, thank you! How can I assist you today?"

    Notes:
        - The function uses a fixed set of generation parameters (max_new_tokens=2048, temperature=0.7, top_p=0.9, etc.).
        - The reply is extracted by removing the prompt portion from the generated sequence.
        - This function is blocking and does not support streaming output. For streaming, use `stream_generate`.
    """
    # Disable gradient computation for inference
    with torch.inference_mode():
        # Perform text generation with specified parameters
        out = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
        )
    # Extract the generated tokens excluding the input prompt
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = out.sequences[0, input_len:]
    reply = tok.decode(generated_ids, skip_special_tokens=True).strip() 
    return reply



def stream_generate(
    inputs: dict,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 4096
) -> Generator[str | None, None, str]:
    """
    Generate text in a streaming fashion, yielding partial output chunks as they are produced by the model.

    This function launches a background thread to run the model's `generate` method with a streaming callback,
    allowing partial text to be yielded as soon as it is available.

    Args:
        inputs (dict):
            Dictionary of model inputs.
        tok (AutoTokenizer):
            The Hugging Face tokenizer instance used for decoding generated tokens.
        model (AutoModelForCausalLM):
            The Hugging Face causal language model instance used for text generation.
        max_new_tokens (int, optional):
            Maximum number of new tokens to generate. Defaults to 2048.

    Yields:
        str: Partial text chunks as they are generated by the model.
        None: A final None value is yielded after generation is complete.

    Returns:
        str: The full concatenated generated text (returned as the generator's return value).

    Example:
        >>> for chunk in stream_generate(inputs, tok, model):
        ...     if chunk is not None:
        ...         print(chunk, end="", flush=True)
        ...

    Notes:
        - This function is a generator; use `yield from` or iterate over it to receive streaming output.
        - The final return value (full text) can be accessed via generator's `return` (Python 3.3+).
        - For non-streaming/blocking generation, use `generate_reply` instead.
    """
    # Set up the streamer to handle token-by-token output
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True, skip_prompt=True)
    # Define generation parameters
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        streamer=streamer,
    )
    # Start generation in a separate thread to enable streaming
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    # Yield chunks as they are produced by the streamer
    chunks = []
    for chunk in streamer:
        chunks.append(chunk)
        yield chunk
    thread.join()
    yield None
    # Return the full concatenated generated text
    return "".join(chunks)

def get_reply(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt_dict: dict,
    message: str,
    role: str = "assistant",
    conversation_history: Optional[list] = None
) -> str:
    
    """
    Generate a full assistant reply in a blocking (non-streaming) manner.

    This function prepares the model inputs using the provided tokenizer, model, system prompt configuration,
    user message, and conversation history, then generates a reply using the language model. It is suitable for
    synchronous applications where the entire response is needed at once.

    Args:
        tok (AutoTokenizer):
            The Hugging Face tokenizer instance used for encoding and decoding text.
        model (AutoModelForCausalLM):
            The Hugging Face causal language model instance used for text generation.
        system_prompt_dict (dict):
            Dictionary containing system prompt configuration.
        message (str):
            The user message or prompt to which the assistant should reply.
        role (str, optional):
            The role for the reply (e.g., "assistant"). Defaults to "assistant".
        conversation_history (Optional[list], optional):
            List of previous conversation turns, if any, to provide context. Defaults to None.

    Returns:
        str: The generated assistant reply as a string.

    Example:
        >>> reply = get_reply(tok, model, system_prompt_dict, "How do I use this API?")
        >>> print(reply)
        "You can use the API by ..."

    Notes:
        - This function is blocking and returns the full reply at once.
        - For streaming output, use `stream_reply` instead.
    """
    # Prepare inputs for the model
    inputs = prepare_inputs(
        tok, 
        model, 
        system_prompt_dict, 
        message, 
        role, conversation_history
    )
    # Generate and return the reply
    return generate_reply(inputs, tok, model)

def stream_reply(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt_dict: dict,
    user_message: str,
    role: str = "assistant",
    conversation_history: Optional[list] = None
) -> Generator[str | None, None, dict]:
    
    """
    Yield the assistant's reply in a streaming fashion, producing partial text chunks as they are generated.

    This function prepares the model inputs and uses a streaming generator to yield partial outputs as soon as they are available.
    It is suitable for interactive applications where incremental output is desired (e.g., chatbots, live UIs).

    Args:
        tok (AutoTokenizer):
            The Hugging Face tokenizer instance used for encoding and decoding text.
        model (AutoModelForCausalLM):
            The Hugging Face causal language model instance used for text generation.
        system_prompt_dict (dict):
            Dictionary containing system prompt configuration and metadata.
        user_message (str):
            The user message or prompt to which the assistant should reply.
        role (str, optional):
            The role for the reply (e.g., "assistant"). Defaults to "assistant".
        conversation_history (Optional[list], optional):
            List of previous conversation turns, if any, to provide context. Defaults to None.

    Yields:
        str: Partial text chunks as they are generated by the model.
        None: A final None value is yielded after generation is complete.

    Returns:
        dict: A dictionary containing the full reply under the key "reply" (returned 
        as the generator's return value).

    Example:
        >>> for chunk in stream_reply(tok, model, system_prompt_dict, "Hello!"):
        ...     if chunk is not None:
        ...         print(chunk, end="", flush=True)
        ...

    Notes:
        - This function is a generator; use `yield from` or iterate over it to receive streaming output.
        - The final return value (full text) can be accessed via generator's `return` (Python 3.3+).
        - For blocking, non-streaming output, use `get_reply` instead.
    """
    # Prepare inputs for the model
    inputs = prepare_inputs(
        tok, 
        model, 
        system_prompt_dict, 
        user_message,
        role, 
        conversation_history
    )
    # Stream the reply
    reply = yield from stream_generate(inputs, tok, model)
    # Finalize the reply
    return {
        "reply": reply
    }

def parse_review_sections(reply: str) -> tuple[str, str, str]:
    """
    Extract the improvements, revised answer, and comments sections from a review reply string.

    This function parses a reply string expected to contain sections marked by
    '### Improvements', '### Revised Answer', and '### Comments'. It extracts the content
    of each section, applies code fence formatting, and returns the results as a tuple.
    If a section is missing, 'None' is returned for that section.

    Args:
        reply (str):
            The full review reply string containing the marked sections.

    Returns:
        tuple[str, str, str]:
            A tuple containing (improvements, revised_answer, comments), each as a string.

    Example:
        >>> improvements, revised, comments = parse_review_sections(reply)
        >>> print(improvements)
        ...

    Notes:
        - Section markers must match exactly (e.g., '### Improvements').
        - If a section is missing, 'None' is returned for that section.
        - Code fence formatting is applied to each section using ensure_fenced_code.
    """
    if "### Improvements" in reply:
        if "### Revised Answer" in reply:
            suggestions = reply.split("### Improvements")[1].split("### Revised Answer")[0].strip()
        else:
            suggestions = reply.split("### Improvements")[1].strip()
    else:
        suggestions = "None"

    if "### Revised Answer" in reply:
        if "### Comments" in reply:
            revised_answer = reply.split("### Revised Answer")[1].split("### Comments")[0].strip()
        else:
            revised_answer = reply.split("### Revised Answer")[1].strip()
    else:
        revised_answer = "None"

    if "### Comments" in reply:
        comments = reply.split("### Comments")[1].strip()
    else:
        comments = "None"
    return (
        suggestions,
        revised_answer,
        comments,
    )

def get_revised_reply(
    user_message: str,
    draft_answer: str,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt_dict: dict,
    role: str = "reviser",
    conversation_history: Optional[list] = None
) -> tuple[str, str, str]:
    """
    Run a single review pass to generate improvements, a revised answer, and comments for a draft response.

    This function builds a review message from the user message and draft answer, then generates a model reply
    and parses it into three sections: improvements, revised answer, and comments. It is typically used as part
    of a multi-step refinement pipeline for LLM outputs.

    Args:
        user_message (str):
            The original user message or question.
        draft_answer (str):
            The initial draft answer to be reviewed and improved.
        tok (AutoTokenizer):
            The Hugging Face tokenizer instance used for encoding and decoding text.
        model (AutoModelForCausalLM):
            The Hugging Face causal language model instance used for text generation.
        system_prompt_dict (dict):
            Dictionary containing system prompt configuration and metadata.
        role (str, optional):
            The role for the review (e.g., "reviser"). Defaults to "reviser".
        conversation_history (Optional[list], optional):
            List of previous conversation turns, if any, to provide context. Defaults to None.

    Returns:
        tuple[str, str, str]:
            A tuple containing (improvements, revised_answer, comments), each as a string.

    Example:
        >>> improvements, revised, comments = get_revised_reply(user_message, draft_answer, tok, model, system_prompt_dict)
        >>> print(revised)
        ...

    Notes:
        - This function is blocking and returns all sections at once.
        - For multi-pass refinement, call this function multiple times with updated answers.
    """
    review_message = build_review_message(user_message, draft_answer)
    reply = get_reply(
        tok, 
        model, 
        system_prompt_dict, 
        review_message, 
        role, 
        conversation_history
    )
    return parse_review_sections(reply)

def get_and_improve_reply(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt_dict: dict,
    user_message: str,
    conversation_history: Optional[list] = None
) -> dict[str, str]:
    """
    Run a two-pass refinement pipeline to generate and improve an assistant's reply.

    This function first generates a draft answer to the user's message, then performs two review passes:
    the first review suggests improvements and produces a revised answer, and the second review further
    refines the answer. All suggestions, revisions, and comments from both passes are returned in a dictionary.

    Args:
        tok (AutoTokenizer):
            The Hugging Face tokenizer instance used for encoding and decoding text.
        model (AutoModelForCausalLM):
            The Hugging Face causal language model instance used for text generation.
        system_prompt_dict (dict):
            Dictionary containing system prompt configuration and metadata.
        user_message (str):
            The user message or prompt to which the assistant should reply.
        conversation_history (Optional[list], optional):
            List of previous conversation turns, if any, to provide context. Defaults to None.

    Returns:
        dict[str, str]:
            A dictionary containing the user question, draft answer, suggestions, revisions, and comments
            from both review passes. Keys include:
                - "user_question"
                - "draft_answer"
                - "suggestions1"
                - "revision1"
                - "comments1"
                - "suggestions2"
                - "revision2"
                - "comments2"

    Example:
        >>> result = get_and_improve_reply(tok, model, system_prompt_dict, "How do I use this API?")
        >>> print(result["revision2"])
        ...

    Notes:
        - This function is blocking and returns all results at once.
        - For streaming output, use a streaming variant of this function.
    """
    draft = get_reply(
        tok, 
        model, 
        system_prompt_dict, 
        user_message, 
        role = "assistant", 
        conversation_history=conversation_history
    )
    suggestions1, revised1, comments1 = get_revised_reply(
        user_message, 
        draft, 
        tok, 
        model, 
        system_prompt_dict, 
        role = "reviser", 
        conversation_history=conversation_history
    )
    suggestions2, revised2, comments2 = get_revised_reply(
        user_message, 
        revised1, 
        tok, 
        model, 
        system_prompt_dict, 
        role = "reviser", 
        conversation_history=conversation_history
    )

    return {
        "user_question": user_message,
        "draft_answer": draft,
        "suggestions1": suggestions1,
        "revision1": revised1,
        "comments1": comments1,
        "suggestions2": suggestions2,
        "revision2": revised2,
        "comments2": comments2
    }

def stream_and_improve_reply(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt_dict: dict,
    user_message: str,
    conversation_history: Optional[list] = None
) -> Generator[str, None, dict[str, str]]:
    """
    Run a two-pass refinement pipeline with streaming output for the final pass.

    This function first generates a draft answer to the user's message, then performs two review passes:
    the first review suggests improvements and produces a revised answer, and the second review further
    refines the answer. The final review is streamed, yielding partial output chunks as they are generated.
    All suggestions, revisions, and comments from both passes are returned in a dictionary.

    Args:
        tok (AutoTokenizer):
            The Hugging Face tokenizer instance used for encoding and decoding text.
        model (AutoModelForCausalLM):
            The Hugging Face causal language model instance used for text generation.
        system_prompt_dict (dict):
            Dictionary containing system prompt configuration and metadata.
        user_message (str):
            The user message or prompt to which the assistant should reply.
        conversation_history (Optional[list], optional):
            List of previous conversation turns, if any, to provide context. Defaults to None.

    Yields:
        str: Partial text chunks or status messages as they are generated by the model.

    Returns:
        dict[str, str]:
            A dictionary containing the user question, draft answer, suggestions, revisions, and comments
            from both review passes. Keys include:
                - "user_question"
                - "draft_answer"
                - "suggestions1"
                - "revision1"
                - "comments1"
                - "suggestions2"
                - "revision2"
                - "comments2"

    Example:
        >>> for chunk in stream_and_improve_reply(tok, model, system_prompt_dict, "How do I use this API?"):
        ...     if chunk is not None:
        ...         print(chunk, end="", flush=True)
        ...

    Notes:
        - This function is a generator; use `yield from` or iterate over it to receive streaming output.
        - The final return value (full text) can be accessed via generator's `return` (Python 3.3+).
        - For blocking, non-streaming output, use `get_and_improve_reply` instead.
    """
    yield("🤔 Generating draft answer...")
    draft = get_reply(
        tok, 
        model, 
        system_prompt_dict, 
        user_message, 
        role = "assistant", 
        conversation_history=conversation_history
    )
    yield("🔄️ Running first improvement pass...")
    suggestions1, revised1, comments1 = get_revised_reply(
        user_message, 
        draft, 
        tok, 
        model, 
        system_prompt_dict, 
        role = "reviser", 
        conversation_history=conversation_history
    )
    review_message2 = build_review_message(user_message, revised1)
    inputs = prepare_inputs(
        tok, 
        model, 
        system_prompt_dict, 
        review_message2, 
        role = "reviser", 
        conversation_history=conversation_history
    )
    yield("🔄️ Running second improvement pass...\n\n")
    reply = yield from stream_generate(inputs,tok,model)
    suggestions2, revised2, comments2 = parse_review_sections(reply)

    return {
        "user_question": user_message,
        "draft_answer": draft,
        "suggestions1": suggestions1,
        "revision1": revised1,
        "comments1": comments1,
        "suggestions2": suggestions2,
        "revision2": revised2,
        "comments2": comments2
    }

def stream_and_improve_reply_display_intermediate(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt_dict: dict,
    user_message: str,
    conversation_history: Optional[list] = None
) -> Generator[str, None, dict[str, str]]:
    """
    Run a multi-stage refinement pipeline with streaming output and display of intermediate results.

    This function streams the draft answer, the first review, and the second review, yielding partial output
    chunks and newlines between each stage. All suggestions, revisions, and comments from both review passes
    are returned in a dictionary.

    Args:
        tok (AutoTokenizer):
            The Hugging Face tokenizer instance used for encoding and decoding text.
        model (AutoModelForCausalLM):
            The Hugging Face causal language model instance used for text generation.
        system_prompt_dict (dict):
            Dictionary containing system prompt configuration and metadata.
        user_message (str):
            The user message or prompt to which the assistant should reply.
        conversation_history (Optional[list], optional):
            List of previous conversation turns, if any, to provide context. Defaults to None.

    Yields:
        str: Partial text chunks or newlines as they are generated by the model.

    Returns:
        dict[str, str]:
            A dictionary containing the user question, draft answer, suggestions, revisions, and comments
            from both review passes. Keys include:
                - "user_question"
                - "draft_answer"
                - "suggestions1"
                - "revision1"
                - "comments1"
                - "suggestions2"
                - "revision2"
                - "comments2"

    Example:
        >>> for chunk in stream_and_improve_reply_display_intermediate(tok, model, system_prompt_dict, "How do I use this API?"):
        ...     if chunk is not None:
        ...         print(chunk, end="", flush=True)
        ...

    Notes:
        - This function is a generator; use `yield from` or iterate over it to receive streaming output.
        - The final return value (full text) can be accessed via generator's `return` (Python 3.3+).
        - For blocking, non-streaming output, use `get_and_improve_reply` instead.
    """
    inputs = prepare_inputs(
        tok, 
        model, 
        system_prompt_dict, 
        user_message, 
        role = "assistant", 
        conversation_history=conversation_history
    )
    draft = yield from stream_generate(inputs,tok,model)
    yield "\n"
    review_message = build_review_message(user_message, draft)
    inputs = prepare_inputs(
        tok, 
        model, 
        system_prompt_dict, 
        review_message, 
        role = "reviser", 
        conversation_history=conversation_history
    )
    reply = yield from stream_generate(inputs,tok,model)
    yield "\n"
    suggestions1, revised1, comments1 = parse_review_sections(reply)
    review_message = build_review_message(user_message, revised1)
    inputs = prepare_inputs(
        tok, 
        model, 
        system_prompt_dict, 
        review_message, 
        role = "reviser", 
        conversation_history=conversation_history
    )
    reply = yield from stream_generate(inputs,tok,model)
    suggestions2, revised2, comments2 = parse_review_sections(reply)

    return {
        "user_question": user_message,
        "draft_answer": draft,
        "suggestions1": suggestions1,
        "revision1": revised1,
        "comments1": comments1,
        "suggestions2": suggestions2,
        "revision2": revised2,
        "comments2": comments2
    }

