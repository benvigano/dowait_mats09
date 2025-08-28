"""
GPU Memory Test Module for RTX A6000
This module provides functions to test GPU memory by loading large language models.
"""

import torch
import time
import psutil
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

def get_gpu_info():
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return None
    
    print(f"✅ CUDA is available!")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return gpu_count

def get_memory_usage():
    """Get current memory usage."""
    # GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"System Memory - Used: {memory.used / (1024**3):.2f} GB, Available: {memory.available / (1024**3):.2f} GB")

def load_model(model_name="google/gemma-2-9b"):
    """Load a large language model to test GPU memory."""
    print(f"\n🚀 Loading model: {model_name}")
    print("This may take a few minutes...")
    
    start_time = time.time()
    
    try:
        # Get HF token from environment
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️  Warning: HF_TOKEN not found in environment variables")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with device map for optimal memory usage
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",  # Automatically handle device placement
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def run_inference(model, tokenizer, prompt, max_new_tokens=50, temperature=0.9, top_p=0.95, do_sample=True, verbose=True):
    """
    Run inference with the loaded model.
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer for the model
        prompt (str): Input text prompt
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature (0.0 to 2.0)
        top_p (float): Nucleus sampling probability threshold
        do_sample (bool): Whether to use sampling or greedy decoding
        verbose (bool): Whether to print progress information
    
    Returns:
        str: Generated text (without input prompt) or None if error
    """
    if verbose:
        print(f"🤖 Running inference with prompt: '{prompt}'")
    
    try:
        # Encode the input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Move to GPU if available
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        if verbose:
            print("Generating response...")
        
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                top_p=top_p if do_sample else None,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        generation_time = time.time() - start_time
        
        # Decode the output
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated text (remove input prompt)
        generated_text = full_response[len(prompt):].strip()
        
        if verbose:
            print(f"✅ Generation completed in {generation_time:.2f} seconds!")
            print(f"Generated text: {generated_text}")
        
        return generated_text
        
    except Exception as e:
        if verbose:
            print(f"❌ Error during inference: {e}")
        raise e

def cleanup_memory(model, tokenizer):
    """Clean up memory after model usage."""
    print("\n🧹 Cleaning up...")
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n📊 Memory Status After Cleanup:")
    get_memory_usage()


