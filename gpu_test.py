#!/usr/bin/env python3
"""
GPU Memory Test Script for RTX A6000
This script loads a large language model and performs inference to test GPU memory utilization.
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

def load_model(model_name="microsoft/DialoGPT-large"):
    """Load a large language model to test GPU memory."""
    print(f"\n🚀 Loading model: {model_name}")
    print("This may take a few minutes...")
    
    start_time = time.time()
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with device map for optimal memory usage
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",  # Automatically handle device placement
            low_cpu_mem_usage=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def run_inference(model, tokenizer, prompt="Hello, world!"):
    """Run inference with the loaded model."""
    print(f"\n🤖 Running inference with prompt: '{prompt}'")
    
    try:
        # Encode the input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        print("Generating response...")
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode the output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ Generation completed in {generation_time:.2f} seconds!")
        print(f"Response: {response}")
        
        return response
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return None

def main():
    """Main function to run the GPU test."""
    print("=" * 60)
    print("🎯 RTX A6000 GPU Memory Test")
    print("=" * 60)
    
    # Check GPU availability
    gpu_count = get_gpu_info()
    if not gpu_count:
        return
    
    print(f"\n📊 Initial Memory Status:")
    get_memory_usage()
    
    # Load model
    model, tokenizer = load_model()
    if model is None:
        return
    
    print(f"\n📊 Memory Status After Model Loading:")
    get_memory_usage()
    
    # Run inference
    response = run_inference(model, tokenizer)
    
    print(f"\n📊 Final Memory Status:")
    get_memory_usage()
    
    # Clean up
    print("\n🧹 Cleaning up...")
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n📊 Memory Status After Cleanup:")
    get_memory_usage()
    
    print("\n" + "=" * 60)
    print("✅ GPU Memory Test Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
