"""
conversation_engine.py

Conversation management engine for LLM-based assistant applications.

This module defines the ConversationEngine class, which orchestrates the flow of conversation
with a language model. It supports streaming and blocking reply generation, multi-pass answer
improvement, intermediate step display, and conversation history management for context-aware
responses. The engine is designed for integration in chatbots, virtual assistants, and other
conversational AI systems using Hugging Face Transformers.

Key Features:
- Streaming and blocking reply generation with optional multi-stage refinement.
- Configurable answer improvement and intermediate step display.
- Automatic conversation history tracking for context.
- Method chaining for configuration.

Typical usage example:
    >>> engine = ConversationEngine(tok, model, system_prompt, improvement=True)
    >>> for chunk in engine.stream_reply('How do I use this API?'):
    ...     print(chunk, end='', flush=True)
    ...
    >>> print(engine.last_result['revision2'])

Author: Andreas Rasmusson
"""


from conversation_engine.outputs import *

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

    Example:
        >>> engine = ConversationEngine(tok, model, system_prompt, improvement=True)
        >>> for chunk in engine.stream_reply('How do I use this API?'):
        ...     print(chunk, end='', flush=True)
        ...
        >>> print(engine.last_result['revision2'])
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
