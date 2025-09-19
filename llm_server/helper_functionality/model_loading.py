"""
model_loading.py
---------------
Utility for loading and initializing quantized large language models (LLMs) and tokenizers for inference.

This module provides a streamlined function to load a Hugging Face-compatible LLM and tokenizer with 4-bit quantization
using BitsAndBytes, optimized for CUDA devices. It is intended for use in conversational AI pipelines and other
applications requiring efficient, high-performance LLM inference.

Main Function:
    - initialize: Loads, quantizes, and compiles the model and tokenizer for immediate use.

Author: Andreas Rasmusson
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import Tuple
#from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration

def initialize(
        device: str,
        model_id: str
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Initialize and return a Hugging Face tokenizer and quantized LLM model for inference.

    Loads the specified model and tokenizer from the Hugging Face Hub, applies 4-bit quantization
    using BitsAndBytes for efficient inference, compiles the model for performance, and sets
    high-precision matrix multiplication. The function is tailored for CUDA-enabled devices and
    returns both the tokenizer and the ready-to-use model.

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]:
            - The loaded tokenizer for text encoding/decoding.
            - The quantized, compiled language model ready for inference on GPU.
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map=device,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model = torch.compile(model)
    torch.set_float32_matmul_precision("high")
    return tok, model
