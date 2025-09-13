"""
Modular model interface supporting both HuggingFace and Nebius AI API.
"""

import os
import torch
from abc import ABC, abstractmethod
from typing import Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from transformer_lens import HookedTransformer

from cache import (
    get_from_generation_cache, save_to_generation_cache,
    print_timestamped_message
)


class ModelInterface(ABC):
    """Abstract interface for language models."""
    
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 4096, 
                 temperature: float = 0.2, top_p: float = 0.75, 
                 do_sample: bool = True) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def get_model_id(self) -> str:
        """Return the model identifier for caching."""
        pass
    
    @abstractmethod
    def get_model_for_patching(self) -> Any:
        """Return a model instance suitable for activation patching."""
        pass
    
    @abstractmethod 
    def get_nnsight_model(self) -> Any:
        """Return a model instance suitable for nnsight operations."""
        pass


class HuggingFaceModel(ModelInterface):
    """HuggingFace transformers implementation."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        print_timestamped_message(f"Loading HuggingFace model '{self.model_id}'...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Native loading for all models - no quantization for better activation analysis
        print_timestamped_message("Loading model natively with bfloat16 precision...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},  # Force everything on GPU 0 for predictable memory management
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print_timestamped_message("HuggingFace model loaded successfully.")
        print_timestamped_message(f"Model device: {next(self.model.parameters()).device}")
    
    def generate(self, prompt: str, max_new_tokens: int = 4096, 
                 temperature: float = 0.2, top_p: float = 0.75, 
                 do_sample: bool = True) -> str:
        """Generate text using HuggingFace model with caching."""
        # Check cache first using normalized model ID
        cache_model_id = self.get_model_id()
        cached_result = get_from_generation_cache(prompt, cache_model_id)
        if cached_result is not None:
            return cached_result
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate with HuggingFace
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Only save to cache if generation was successful (no errors)
        if not generated_text.startswith("Error"):
            save_to_generation_cache(prompt, generated_text, cache_model_id)
        
        return generated_text
    
    def get_model_id(self) -> str:
        # Normalize Qwen model IDs to match Nebius cache keys
        if "qwen3" in self.model_id.lower() or "qwen/qwen3" in self.model_id.lower():
            return "nebius-Qwen/Qwen3-14B"  # Match the Nebius cache format
        return self.model_id
    
    def get_model_for_patching(self) -> HookedTransformer:
        """
        Return a HookedTransformer instance for patching.
        This will drop the original HuggingFace model to free memory and load a new
        one directly through TransformerLens to avoid device map issues.
        """
        if hasattr(self, 'hooked_model') and self.hooked_model is not None:
            return self.hooked_model

        # Drop the original HF model (which might be on multiple devices)
        if self.model is not None:
            print_timestamped_message("Dropping original HuggingFace model to prepare for patching...")
            self.model = drop_model_from_memory(self.model)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print_timestamped_message(f"Loading HookedTransformer directly for '{self.model_id}' on device '{device}'...")
        
        # Load HookedTransformer directly without passing an hf_model
        # This should avoid the device_map issues entirely
        self.hooked_model = HookedTransformer.from_pretrained(
            self.model_id,
            device=device,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16
        )

        print_timestamped_message("✅ HookedTransformer loaded directly.")
        return self.hooked_model
    
    def get_nnsight_model(self) -> Any:
        """Return model for nnsight operations. Reloads with int4 quantization if needed."""
        try:
            import nnsight
            
            # Print memory status before loading
            if torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated() / 1024**3
                print_timestamped_message(f"GPU memory before loading: {gpu_memory_before:.2f}GB")
            
            # Always reload the model for nnsight to ensure clean state
            print_timestamped_message("Loading fresh model instance for nnsight operations...")
            if self.model is not None:
                # Clear existing model first
                self.model = drop_model_from_memory(self.model)
            
            # Reload model (will use int4 quantization for Qwen3)
            self._load_model()
            
            # Print memory status after loading
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**3
                print_timestamped_message(f"GPU memory after loading: {gpu_memory_after:.2f}GB")
            
            # Wrap with nnsight - this allows us to trace activations
            nnsight_model = nnsight.LanguageModel(self.model, tokenizer=self.tokenizer)
            return nnsight_model
        except ImportError:
            print_timestamped_message("⚠️ nnsight not available. Install with: pip install nnsight")
            return None
        except Exception as e:
            print_timestamped_message(f"⚠️ Failed to create nnsight model: {e}")
            return None


class NebiusModel(ModelInterface):
    """Nebius AI API implementation using OpenAI client."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the OpenAI client for Nebius AI."""
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY environment variable not set")
        
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=api_key
        )
        
        print_timestamped_message(f"Nebius AI client setup complete for model '{self.model_id}'")
    
    def generate(self, prompt: str, max_new_tokens: int = 4096, 
                 temperature: float = 0.2, top_p: float = 0.75, 
                 do_sample: bool = True) -> str:
        """Generate text using Nebius AI API with caching."""
        # Check cache first (use Nebius- prefix to distinguish from HuggingFace cache)
        cache_model_id = f"nebius-{self.model_id}"
        cached_result = get_from_generation_cache(prompt, cache_model_id)
        if cached_result is not None:
            return cached_result
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0.0,
                top_p=top_p if do_sample else 1.0,
            )
            
            generated_text = response.choices[0].message.content
            
            # Save to cache with Nebius prefix
            if generated_text and not generated_text.startswith("Error"):
                save_to_generation_cache(prompt, generated_text, cache_model_id)
            
            return generated_text
            
        except Exception as e:
            error_msg = f"Error generating with Nebius API: {str(e)}"
            print_timestamped_message(error_msg)
            return error_msg
    
    def get_model_id(self) -> str:
        return f"nebius-{self.model_id}"
    
    def get_model_for_patching(self) -> Any:
        """Activation patching is not supported for API-based models."""
        print_timestamped_message("⚠️ Activation patching not supported for NebiusModel.")
        return None
    
    def get_nnsight_model(self) -> Any:
        """NNsight is not supported for API-based models."""
        print_timestamped_message("⚠️ NNsight not supported for NebiusModel.")
        return None


class HybridQwen3Model(ModelInterface):
    """
    Hybrid model that uses Nebius API for inference but supports TransformerLens for patching.
    Specifically designed for Qwen3 models that are available both via Nebius API and HuggingFace.
    """
    
    def __init__(self, nebius_model_id: str, hf_model_id: str):
        self.nebius_model_id = nebius_model_id
        self.hf_model_id = hf_model_id
        self.nebius_model = NebiusModel(nebius_model_id)
        self.hf_tokenizer = None
        self.hooked_model = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the HuggingFace tokenizer for patching operations."""
        try:
            print_timestamped_message(f"Loading tokenizer for '{self.hf_model_id}'...")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            print_timestamped_message("✅ Tokenizer loaded successfully")
        except Exception as e:
            print_timestamped_message(f"⚠️ Failed to load tokenizer: {e}")
            self.hf_tokenizer = None
    
    def generate(self, prompt: str, max_new_tokens: int = 4096, 
                 temperature: float = 0.2, top_p: float = 0.75, 
                 do_sample: bool = True) -> str:
        """Generate text using Nebius API (faster and more efficient)."""
        return self.nebius_model.generate(prompt, max_new_tokens, temperature, top_p, do_sample)
    
    def get_model_id(self) -> str:
        """Return the Nebius model ID for caching consistency."""
        return self.nebius_model.get_model_id()
    
    def get_model_for_patching(self) -> HookedTransformer:
        """
        Return a HookedTransformer instance for patching using the HuggingFace model.
        This loads the HF version of the model for TransformerLens compatibility.
        """
        if hasattr(self, 'hooked_model') and self.hooked_model is not None:
            return self.hooked_model

        if self.hf_tokenizer is None:
            print_timestamped_message("⚠️ Cannot load HookedTransformer without tokenizer")
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print_timestamped_message(f"Loading HookedTransformer for '{self.hf_model_id}' on device '{device}'...")
        
        try:
            # Load HookedTransformer directly for the HuggingFace version
            self.hooked_model = HookedTransformer.from_pretrained(
                self.hf_model_id,
                device=device,
                tokenizer=self.hf_tokenizer,
                torch_dtype=torch.bfloat16
            )
            print_timestamped_message("✅ HookedTransformer loaded successfully for patching")
            return self.hooked_model
            
        except Exception as e:
            print_timestamped_message(f"⚠️ Failed to load HookedTransformer: {e}")
            return None
    
    def get_nnsight_model(self) -> Any:
        """Return nnsight model for activation steering (uses HuggingFace model)."""
        try:
            import nnsight
            
            if self.hf_tokenizer is None:
                print_timestamped_message("⚠️ Cannot create nnsight model without tokenizer")
                return None
            
            # Load the HuggingFace model for nnsight
            print_timestamped_message(f"Loading HuggingFace model for nnsight: '{self.hf_model_id}'...")
            
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Wrap with nnsight
            nnsight_model = nnsight.LanguageModel(hf_model, tokenizer=self.hf_tokenizer)
            
            print_timestamped_message("✅ NNsight model loaded successfully")
            return nnsight_model
            
        except ImportError:
            print_timestamped_message("⚠️ nnsight not available. Install with: pip install nnsight")
            return None
        except Exception as e:
            print_timestamped_message(f"⚠️ Failed to create nnsight model: {e}")
            return None


# Removed: TransformerLensModel class


def create_model(model_type: str, model_id: str) -> ModelInterface:
    """Factory function to create the appropriate model implementation."""
    if model_type.lower() == "huggingface":
        return HuggingFaceModel(model_id)
    elif model_type.lower() == "nebius":
        return NebiusModel(model_id)
    elif model_type.lower() == "hybrid_qwen3":
        # For hybrid Qwen3, model_id should be in format "nebius_id|hf_id"
        if "|" not in model_id:
            raise ValueError("Hybrid Qwen3 model_id must be in format 'nebius_id|hf_id'")
        nebius_id, hf_id = model_id.split("|", 1)
        return HybridQwen3Model(nebius_id, hf_id)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'huggingface', 'nebius', or 'hybrid_qwen3'.")


def drop_model_from_memory(model):
    """Aggressively drop a model from memory to prevent OOM."""
    if model is not None:
        print_timestamped_message("🗑️ Dropping model from memory to prevent OOM...")
        
        # If it's our model interface, get the underlying model
        if hasattr(model, 'model'):
            underlying_model = model.model
        else:
            underlying_model = model
        
        # Move to CPU and delete
        if hasattr(underlying_model, 'cpu'):
            underlying_model.cpu()
        del underlying_model
        
        # Aggressive memory cleanup
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # More aggressive cleanup
            torch.cuda.synchronize()
            
            # Multiple cleanup passes
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
        
        print_timestamped_message("✅ Model dropped from memory")
        return None
    return model


