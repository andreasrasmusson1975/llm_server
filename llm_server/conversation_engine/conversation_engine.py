"""
Conversation Engine Module

This module provides the core conversational AI engine for the LLM Server, implementing
sophisticated conversation management with optional multi-pass improvement capabilities.

The ConversationEngine class serves as the primary interface for managing interactions
with Large Language Models, providing both streaming and blocking response generation
with advanced features like conversation history management, response improvement,
and intermediate step transparency.

Key Features:
    - Streaming Response Generation: Real-time token-by-token output streaming
    - Conversation History: Automatic context management across multiple turns
    - Multi-Pass Improvement: Optional response refinement and revision
    - Intermediate Steps: Transparent display of improvement process
    - Method Chaining: Fluent interface for configuration
    - Memory Management: Efficient handling of conversation context

Architecture:
    The engine acts as a high-level orchestrator that delegates specific generation
    tasks to specialized functions in the outputs module. It manages the conversation
    flow, history tracking, and optional improvement pipeline while providing a
    clean, consistent interface for the API layer.

Response Generation Modes:
    1. Basic Mode: Direct streaming generation without improvement
    2. Improvement Mode: Multi-pass generation with response refinement
    3. Intermediate Mode: Improvement with visible intermediate steps

Conversation Flow:
    1. User message received
    2. Context preparation (history + system prompt)
    3. Model generation (streaming or blocking)
    4. Optional improvement passes
    5. History update
    6. Result storage

Dependencies:
    - transformers: AutoTokenizer, AutoModelForCausalLM
    - llm_server.conversation_engine.outputs: Core generation functions
    - typing: Generator type hints

Author: Andreas Rasmusson
"""

from llm_server.conversation_engine.outputs import *

class ConversationEngine:
    """
    High-level conversational engine for managing LLM-based assistant interactions.

    This class orchestrates the flow of conversation with a language model, supporting both
    basic and multi-stage improvement pipelines. It manages conversation history, streaming
    output, and optional answer refinement with intermediate steps.

    Features:
        - Streaming and blocking reply generation.
        - Optional multi-pass answer improvement and review.
        - Conversation history management for context-aware responses.
        - Configurable intermediate step display for transparency.

    Args:
        tok: Hugging Face tokenizer instance for encoding/decoding.
        model: Hugging Face causal language model instance for text generation.
        system_prompt: Dictionary containing system prompt configuration and metadata.
        improvement (bool, optional): If True, enables multi-pass answer improvement. Defaults to False.
        intermediate_steps (bool, optional): If True, streams and displays intermediate review steps. Defaults to False.

    Attributes:
        tok: The tokenizer used for encoding/decoding.
        model: The language model used for generation.
        system_prompt: The system prompt configuration.
        conversation_history: List of conversation turns for context.
        improvement: Whether multi-pass improvement is enabled.
        intermediate_steps: Whether to display intermediate review steps.
        last_result: The result of the most recent reply generation.
    """
    def __init__(
        self,
        tok: AutoTokenizer,
        model: AutoModelForCausalLM,
        system_prompt: dict,
        improvement: bool = False,
        intermediate_steps: bool = False
    ) -> None:
        self.tok = tok 
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.improvement = improvement
        self.intermediate_steps = intermediate_steps
        self.last_result = {}

    def set_improvement(self, b: bool) -> "ConversationEngine":
        """
        Enable or disable multi-pass answer improvement for the conversation engine.

        Args:
            b (bool): If True, enables answer improvement; if False, disables it.

        Returns:
            ConversationEngine: Returns self to allow method chaining.

        Example:
            >>> engine.set_improvement(True)
            >>> engine.set_improvement(False)
        """
        self.improvement = b
        return self

    def set_intermediate_steps(self, b: bool) -> "ConversationEngine":
        """
        Enable or disable streaming and display of intermediate review steps.

        Args:
            b (bool): If True, enables streaming of intermediate steps; if False, disables it.

        Returns:
            ConversationEngine: Returns self to allow method chaining.

        Example:
            >>> engine.set_intermediate_steps(True)
            >>> engine.set_intermediate_steps(False)
        """
        self.intermediate_steps = b
        return self

    def stream_reply(self, user_message: str) -> Generator[str, None, None]:
        """
        Generate a streaming reply to a user message, yielding output chunks as they are produced.

        This method manages the reply generation pipeline, optionally enabling multi-pass improvement
        and intermediate step display. It updates the conversation history and stores the final result
        in self.last_result.

        Args:
            user_message (str): The user's input message to the assistant.

        Yields:
            str: Partial output chunks as they are generated by the model.

        Side Effects:
            - Updates self.last_result with the final reply or improved answer.
            - Appends the user and assistant turns to self.conversation_history.

        Example:
            >>> for chunk in engine.stream_reply('How do I use this API?'):
            ...     print(chunk, end='', flush=True)
            ...
            >>> print(engine.last_result['revision2'])
        """
        if self.improvement:
            if self.intermediate_steps:
                gen = stream_and_improve_reply_display_intermediate(
                    tok=self.tok,
                    model=self.model,
                    system_prompt_dict=self.system_prompt,
                    user_message=user_message,
                    conversation_history=self.conversation_history
                )
            else:
                gen = stream_and_improve_reply(
                    tok=self.tok,
                    model=self.model,
                    system_prompt_dict=self.system_prompt,
                    user_message=user_message,
                    conversation_history=self.conversation_history
                )
        else:
            gen = stream_reply(
                tok=self.tok,
                model=self.model,
                system_prompt_dict=self.system_prompt,
                user_message=user_message,
                conversation_history=self.conversation_history,
            )

        # iterate manually so we can catch StopIteration
        while True:
            try:
                chunk = next(gen)
                if chunk is not None:
                    yield chunk
            except StopIteration as e:
                result = e.value   # ✅ final return from inner generator
                self.last_result = result

                # update conversation history
                self.conversation_history.append({"role": "user", "content": user_message})
                if self.improvement:
                    self.conversation_history.append(
                        {"role": "assistant", "content": result["revision2"]}
                    )
                else:
                    self.conversation_history.append(
                        {"role": "assistant", "content": result["reply"]}
                    )
                break
