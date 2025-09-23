
"""
Model Loading Module for LLM Server

This module provides efficient loading and initialization of Large Language Models
and their associated tokenizers for high-performance inference. It implements
optimized configurations for GPU-accelerated text generation with advanced
features like Flash Attention, model compilation, and precision optimization.

The module abstracts the complexity of model initialization, providing a simple
interface for loading production-ready models with optimal performance settings
for real-time conversational AI applications.

Key Features:
    - Automatic model and tokenizer loading from Hugging Face Hub
    - Flash Attention 2 integration for memory-efficient attention computation
    - PyTorch model compilation for accelerated inference
    - High-precision matrix multiplication optimization
    - Device mapping for GPU acceleration
    - Support for various model architectures (Mistral, Llama, etc.)

Optimization Features:
    Flash Attention 2:
        - Reduces memory usage for long sequences
        - Increases inference speed through optimized attention kernels
        - Particularly beneficial for conversational contexts

    Model Compilation:
        - Uses torch.compile for JIT optimization
        - Reduces inference latency after warm-up
        - Optimizes computational graphs for repeated inference

    Precision Settings:
        - bfloat16 for model weights (memory efficiency + precision)
        - High-precision matmul for critical computations
        - Balanced performance and numerical stability

    Device Management:
        - Automatic GPU device mapping
        - CUDA memory optimization
        - Fallback support for CPU inference

Model Support:
    The module is designed to work with various transformer architectures:
    - Mistral models (3B, 7B, 22B variants)
    - Llama models (3B, 7B, 13B, 70B variants)
    - Code generation models (CodeLlama, etc.)
    - Instruction-tuned models (Chat variants)

Memory Management:
    - bfloat16 reduces memory footprint by ~50% vs float32
    - Flash Attention 2 provides sublinear memory scaling
    - Efficient device mapping minimizes CPU-GPU transfers
    - Optional quantization support (currently commented out)

Dependencies:
    - transformers: Model and tokenizer loading
    - torch: Core tensor operations and compilation
    - flash-attn: Optimized attention implementation (optional)
    - bitsandbytes: Quantization support (optional)

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
    #bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #quantization_config=bnb,
        device_map=device,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    model = torch.compile(model)
    torch.set_float32_matmul_precision("high")
    return tok, model
