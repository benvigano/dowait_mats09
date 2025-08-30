
# --- Activation Patching Logic ---
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
import re
from tqdm.notebook import tqdm
import pandas as pd
from datetime import datetime
import pytz
import traceback
import time
import hashlib
import csv
import os

# This file will contain the core logic for our experiments, 
# keeping the main notebook clean and focused on the narrative.

# --- Experiment Configuration ---
# Set to True for a quick run on a small sample, False for the full overnight run.
DEBUG_MODE = False
SAMPLE_SIZE = 10 # Number of examples to use in debug mode.
PROD_SAMPLE_SIZE = 10 # Limit the number of examples for the full run

# --- Global Variables & Constants ---
# It's good practice to define constants that might be used across functions.
# We can initialize them as None and set them from the notebook.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Intervention Phrases ---
# Additional corrective phrases for future experiments (currently unused)
UNUSED_INTERVENTIONS = {
    "Corrective_Strong": " Wait, I made a mistake.",
    "Corrective_Alert": " Hold on!",
    "Corrective_Reconsider": " Wait, I need to reconsider.",
    "Corrective_Wrong": " Wait, that was wrong!",
}

# Current experiment: Only testing the original "Wait" intervention
CURRENT_INTERVENTIONS = {
    "Corrective_Original": " Wait. "
}
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# --- Cache System ---
CACHE_DIR = "cache"
GENERATION_CACHE_FILE = os.path.join(CACHE_DIR, "generation_cache.csv")
ERROR_DETECTION_CACHE_FILE = os.path.join(CACHE_DIR, "error_detection_cache.csv")
_generation_cache = None  # In-memory cache
_error_detection_cache = None  # In-memory cache

# --- Utility Functions ---

def load_model_and_tokenizer(model_id=MODEL_ID):
    """
    Loads the HuggingFace model and tokenizer for generation.
    """
    print_timestamped_message(f"Loading HuggingFace model '{model_id}' and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model for generation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=None,  # Disable auto device mapping
        torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
        trust_remote_code=True
    )
    model.to(DEVICE)
    
    print_timestamped_message("HuggingFace model and tokenizer loaded successfully.")
    return model, tokenizer

def load_tl_model(hf_model, model_id=MODEL_ID):
    """
    Loads the same model into TransformerLens for activation patching.
    """
    print_timestamped_message(f"Loading model '{model_id}' into TransformerLens...")
    
    # Suppress verbose warnings from TransformerLens
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        
        try:
            # Try the standard, high-level loading method first
            tl_model = HookedTransformer.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device=DEVICE,
                trust_remote_code=True
            )
        except RuntimeError as e:
            if "device" in str(e).lower():
                print_timestamped_message("Device error encountered. Trying with from_pretrained_no_processing...")
                # Fallback for device map issues
                tl_model, _ = HookedTransformer.from_pretrained_no_processing(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device=DEVICE,
                    trust_remote_code=True
                )
            else:
                raise e # Re-raise other runtime errors
    
    print_timestamped_message("TransformerLens model loaded successfully.")
    return tl_model


def get_timestamp_in_rome():
    """Returns the current timestamp in 'Europe/Rome' timezone."""
    rome_tz = pytz.timezone('Europe/Rome')
    return datetime.now(rome_tz).strftime('%Y-%m-%d %H:%M:%S %Z')

def print_timestamped_message(message):
    """Prints a message with a Rome timestamp."""
    print(f"[{get_timestamp_in_rome()}] {message}")

# --- Generation Cache Functions ---

def _get_cache_key(problem_text, model_id=None):
    """Generate a deterministic cache key for a problem."""
    # Include model ID in key to avoid cross-model contamination
    model_id = model_id or MODEL_ID
    key_string = f"{model_id}::{problem_text}"
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def _load_generation_cache():
    """Load the generation cache from CSV file."""
    global _generation_cache
    
    if _generation_cache is not None:
        return _generation_cache
        
    _generation_cache = {}
    
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Load existing cache if file exists
    if os.path.exists(GENERATION_CACHE_FILE):
        try:
            with open(GENERATION_CACHE_FILE, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cache_key = row['cache_key']
                    _generation_cache[cache_key] = {
                        'problem_text': row['problem_text'],
                        'full_solution': row['full_solution'],
                        'extracted_answer': row['extracted_answer'],
                        'timestamp': row['timestamp'],
                        'model_id': row['model_id']
                    }
            print_timestamped_message(f"Loaded {len(_generation_cache)} cached generations from {GENERATION_CACHE_FILE}")
        except Exception as e:
            print_timestamped_message(f"Warning: Could not load cache file: {e}")
            _generation_cache = {}
    else:
        print_timestamped_message(f"No existing cache found, starting fresh cache at {GENERATION_CACHE_FILE}")
    
    return _generation_cache

def _save_to_cache(problem_text, full_solution, extracted_answer, model_id=None):
    """Save a generation result to the cache."""
    cache_key = _get_cache_key(problem_text, model_id)
    model_id = model_id or MODEL_ID
    
    # Update in-memory cache
    cache = _load_generation_cache()
    cache[cache_key] = {
        'problem_text': problem_text,
        'full_solution': full_solution,
        'extracted_answer': extracted_answer,
        'timestamp': get_timestamp_in_rome(),
        'model_id': model_id
    }
    
    # Append to CSV file
    file_exists = os.path.exists(GENERATION_CACHE_FILE)
    try:
        with open(GENERATION_CACHE_FILE, 'a', encoding='utf-8', newline='') as f:
            fieldnames = ['cache_key', 'problem_text', 'full_solution', 'extracted_answer', 'timestamp', 'model_id']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'cache_key': cache_key,
                'problem_text': problem_text,
                'full_solution': full_solution,
                'extracted_answer': extracted_answer,
                'timestamp': get_timestamp_in_rome(),
                'model_id': model_id
            })
    except Exception as e:
        print_timestamped_message(f"Warning: Could not save to cache: {e}")

def _get_from_cache(problem_text, model_id=None):
    """Retrieve a generation result from the cache."""
    cache_key = _get_cache_key(problem_text, model_id)
    cache = _load_generation_cache()
    
    if cache_key in cache:
        cached_result = cache[cache_key]
        # Cache hit - suppressed individual logging for cleaner output
        return cached_result['full_solution'], cached_result['extracted_answer']
    
    return None, None

# --- Error Detection Cache Functions ---

def _get_error_cache_key(problem_text, incorrect_cot):
    """Generate a deterministic cache key for error detection."""
    key_string = f"error_detection::{problem_text}::{incorrect_cot}"
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def _load_error_detection_cache():
    """Load the error detection cache from CSV file."""
    global _error_detection_cache
    
    if _error_detection_cache is not None:
        return _error_detection_cache
        
    _error_detection_cache = {}
    
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Load existing cache if file exists
    if os.path.exists(ERROR_DETECTION_CACHE_FILE):
        try:
            with open(ERROR_DETECTION_CACHE_FILE, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cache_key = row['cache_key']
                    _error_detection_cache[cache_key] = {
                        'problem_text': row['problem_text'],
                        'incorrect_cot': row['incorrect_cot'],
                        'error_sentence': row['error_sentence'],
                        'timestamp': row['timestamp'],
                        'api_model': row['api_model']
                    }
            print_timestamped_message(f"Loaded {len(_error_detection_cache)} cached error detections from {ERROR_DETECTION_CACHE_FILE}")
        except Exception as e:
            print_timestamped_message(f"Warning: Could not load error detection cache: {e}")
            _error_detection_cache = {}
    else:
        print_timestamped_message(f"No existing error detection cache found")
    
    return _error_detection_cache

def _save_to_error_cache(problem_text, incorrect_cot, error_sentence, api_model="claude-3-5-sonnet-20240620"):
    """Save an error detection result to the cache."""
    cache_key = _get_error_cache_key(problem_text, incorrect_cot)
    
    # Update in-memory cache
    cache = _load_error_detection_cache()
    cache[cache_key] = {
        'problem_text': problem_text,
        'incorrect_cot': incorrect_cot,
        'error_sentence': error_sentence,
        'timestamp': get_timestamp_in_rome(),
        'api_model': api_model
    }
    
    # Append to CSV file
    file_exists = os.path.exists(ERROR_DETECTION_CACHE_FILE)
    try:
        with open(ERROR_DETECTION_CACHE_FILE, 'a', encoding='utf-8', newline='') as f:
            fieldnames = ['cache_key', 'problem_text', 'incorrect_cot', 'error_sentence', 'timestamp', 'api_model']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'cache_key': cache_key,
                'problem_text': problem_text,
                'incorrect_cot': incorrect_cot,
                'error_sentence': error_sentence,
                'timestamp': get_timestamp_in_rome(),
                'api_model': api_model
            })
    except Exception as e:
        print_timestamped_message(f"Warning: Could not save to error detection cache: {e}")

def _get_from_error_cache(problem_text, incorrect_cot):
    """Retrieve an error detection result from the cache."""
    cache_key = _get_error_cache_key(problem_text, incorrect_cot)
    cache = _load_error_detection_cache()
    
    if cache_key in cache:
        cached_result = cache[cache_key]
        # Error detection cache hit - suppressed individual logging for cleaner output
        return cached_result['error_sentence']
    
    return None

def _get_from_error_cache_by_key(cache_key):
    """Retrieve an error detection result by cache key."""
    cache = _load_error_detection_cache()
    
    if cache_key in cache:
        cached_result = cache[cache_key]
        return cached_result['error_sentence']
    
    return None

def _save_to_error_cache_by_key(cache_key, error_sentence):
    """Save an error detection result by cache key."""
    # Update in-memory cache
    cache = _load_error_detection_cache()
    cache[cache_key] = {
        'error_sentence': error_sentence,
        'timestamp': get_timestamp_in_rome()
    }
    
    # Append to CSV file
    try:
        with open(ERROR_DETECTION_CACHE_FILE, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['cache_key', 'error_sentence', 'timestamp'])
            writer.writerow({
                'cache_key': cache_key,
                'error_sentence': error_sentence,
                'timestamp': get_timestamp_in_rome()
            })
    except Exception as e:
        print_timestamped_message(f"Could not save to error detection cache: {e}")

def prepopulate_cache_from_excel_files():
    """Pre-populate the cache with existing baseline AND intervention results from all Excel files."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print_timestamped_message("No results directory found, skipping cache pre-population")
        return
    
    total_baseline_added = 0
    total_intervention_added = 0
    excel_files = [f for f in os.listdir(results_dir) if f.endswith('.xlsx')]
    
    print_timestamped_message(f"Pre-populating cache from {len(excel_files)} Excel files...")
    
    for excel_file in excel_files:
        excel_path = os.path.join(results_dir, excel_file)
        try:
            xl_file = pd.ExcelFile(excel_path)
            
            # 1. Process Baseline Results
            if 'Baseline_Results' in xl_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name='Baseline_Results')
                valid_entries = df.dropna(subset=['problem', 'full_generated_solution'])
                
                for _, row in valid_entries.iterrows():
                    problem_text = row['problem']
                    full_solution = row['full_generated_solution']
                    
                    # Get extracted answer (prefer generated_answer if available)
                    if 'generated_answer' in df.columns and pd.notna(row['generated_answer']):
                        extracted_answer = row['generated_answer']
                    else:
                        extracted_answer = extract_boxed_answer(full_solution)
                    
                    # Check if this is already cached
                    cached_solution, _ = _get_from_cache(problem_text)
                    if cached_solution is None:
                        # Add to cache (without printing cache hit message)
                        cache_key = _get_cache_key(problem_text)
                        cache = _load_generation_cache()
                        cache[cache_key] = {
                            'problem_text': problem_text,
                            'full_solution': full_solution,
                            'extracted_answer': extracted_answer,
                            'timestamp': f"baseline from {excel_file}",
                            'model_id': MODEL_ID
                        }
                        total_baseline_added += 1
            
            # 2. Process Intervention Results (Control_Experiment or Insertion_Test)
            intervention_sheet = None
            if 'Insertion_Test' in xl_file.sheet_names:
                intervention_sheet = 'Insertion_Test'
            elif 'Control_Experiment' in xl_file.sheet_names:
                intervention_sheet = 'Control_Experiment'
            
            if intervention_sheet:
                df_intervention = pd.read_excel(excel_path, sheet_name=intervention_sheet)
                
                # Check if we have the enhanced traceability fields
                if all(col in df_intervention.columns for col in ['intervention_phrase', 'truncated_cot_length', 'intervened_cot']):
                    # Enhanced format - we can reconstruct exact prompts
                    for _, row in df_intervention.dropna(subset=['intervened_cot']).iterrows():
                        # The prompt that was used for generation
                        if pd.notna(row.get('error_sentence')):
                            # Reconstruct the prompt used: truncated_cot + intervention_phrase
                            problem = row['problem']
                            original_cot = row.get('full_generated_solution', '')  # This might be in a joined table
                            error_sentence = row['error_sentence']
                            intervention_phrase = row['intervention_phrase']
                            
                            if original_cot and error_sentence in original_cot:
                                error_index = original_cot.find(error_sentence)
                                truncated_cot = original_cot[:error_index]
                                prompt_used = truncated_cot + intervention_phrase
                                
                                # Check if this intervention prompt is already cached
                                cached_solution, _ = _get_from_cache(prompt_used)
                                if cached_solution is None:
                                    intervened_cot = row['intervened_cot']
                                    intervened_answer = row.get('intervened_answer', extract_boxed_answer(intervened_cot))
                                    
                                    cache_key = _get_cache_key(prompt_used)
                                    cache = _load_generation_cache()
                                    cache[cache_key] = {
                                        'problem_text': prompt_used,
                                        'full_solution': intervened_cot,
                                        'extracted_answer': intervened_answer,
                                        'timestamp': f"intervention from {excel_file}",
                                        'model_id': MODEL_ID
                                    }
                                    total_intervention_added += 1
                else:
                    # Legacy format - limited intervention caching possible
                    print_timestamped_message(f"Legacy format in {excel_file}, skipping intervention cache population")
                
            print_timestamped_message(f"Processed {excel_file}: baseline={total_baseline_added}, interventions={total_intervention_added}")
                
        except Exception as e:
            print_timestamped_message(f"Could not process {excel_file}: {e}")
    
    total_added = total_baseline_added + total_intervention_added
    if total_added > 0:
        # Save the updated cache to file
        try:
            with open(GENERATION_CACHE_FILE, 'w', encoding='utf-8', newline='') as f:
                fieldnames = ['cache_key', 'problem_text', 'full_solution', 'extracted_answer', 'timestamp', 'model_id']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                cache = _load_generation_cache()
                for cache_key, entry in cache.items():
                    writer.writerow({
                        'cache_key': cache_key,
                        'problem_text': entry['problem_text'],
                        'full_solution': entry['full_solution'],
                        'extracted_answer': entry['extracted_answer'],
                        'timestamp': entry['timestamp'],
                        'model_id': entry['model_id']
                    })
            print_timestamped_message(f"Pre-populated cache with {total_baseline_added} baseline + {total_intervention_added} intervention = {total_added} total entries")
        except Exception as e:
            print_timestamped_message(f"Could not save pre-populated cache: {e}")
    else:
        print_timestamped_message("No new entries to add to cache from Excel files")

def prepopulate_cache_from_excel_files_enhanced():
    """Enhanced pre-population that handles legacy Excel format by cross-referencing Error_Analysis."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print_timestamped_message("No results directory found, skipping cache pre-population")
        return
    
    total_baseline_added = 0
    total_intervention_added = 0
    excel_files = [f for f in os.listdir(results_dir) if f.endswith('.xlsx')]
    
    # Use global intervention phrases constants
    intervention_phrases = CURRENT_INTERVENTIONS
    
    print_timestamped_message(f"Pre-populating cache from {len(excel_files)} Excel files...")
    
    for i, excel_file in enumerate(excel_files, 1):
        print_timestamped_message(f"Processing file {i}/{len(excel_files)}: {excel_file}")
        excel_path = os.path.join(results_dir, excel_file)
        try:
            xl_file = pd.ExcelFile(excel_path)
            baseline_added = 0
            intervention_added = 0
            
            # 1. Process Baseline Results
            if 'Baseline_Results' in xl_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name='Baseline_Results')
                valid_entries = df.dropna(subset=['problem', 'full_generated_solution'])
                
                for _, row in valid_entries.iterrows():
                    problem_text = row['problem']
                    full_solution = row['full_generated_solution']
                    
                    # Get extracted answer (prefer generated_answer if available)
                    if 'generated_answer' in df.columns and pd.notna(row['generated_answer']):
                        extracted_answer = row['generated_answer']
                    else:
                        extracted_answer = extract_boxed_answer(full_solution)
                    
                    # Check if this is already cached
                    cached_solution, _ = _get_from_cache(problem_text)
                    if cached_solution is None:
                        # Add to cache
                        cache_key = _get_cache_key(problem_text)
                        cache = _load_generation_cache()
                        cache[cache_key] = {
                            'problem_text': problem_text,
                            'full_solution': full_solution,
                            'extracted_answer': extracted_answer,
                            'timestamp': f"baseline from {excel_file}",
                            'model_id': MODEL_ID
                        }
                        baseline_added += 1
            
            # 2. Process Intervention Results using Error_Analysis cross-reference
            if 'Control_Experiment' in xl_file.sheet_names and 'Error_Analysis' in xl_file.sheet_names:
                df_intervention = pd.read_excel(excel_path, sheet_name='Control_Experiment')
                df_error = pd.read_excel(excel_path, sheet_name='Error_Analysis')
                
                # Create a lookup for error analysis by problem_id
                error_lookup = {}
                for _, row in df_error.iterrows():
                    if pd.notna(row.get('error_sentence')) and pd.notna(row.get('full_generated_solution')):
                        error_lookup[row['problem_id']] = {
                            'original_cot': row['full_generated_solution'],
                            'error_sentence': row['error_sentence']
                        }
                
                # Process interventions
                for _, row in df_intervention.dropna(subset=['intervened_cot']).iterrows():
                    problem_id = row['problem_id']
                    condition = row['condition']
                    intervened_cot = row['intervened_cot']
                    intervened_answer = row.get('intervened_answer', extract_boxed_answer(intervened_cot))
                    
                    # Get intervention phrase
                    intervention_phrase = intervention_phrases.get(condition, f" {condition}")
                    
                    # Get original CoT and error sentence from error analysis
                    if problem_id in error_lookup:
                        original_cot = error_lookup[problem_id]['original_cot']
                        error_sentence = error_lookup[problem_id]['error_sentence']
                        
                        # Find error location and reconstruct prompt
                        if error_sentence in original_cot:
                            error_index = original_cot.find(error_sentence)
                            truncated_cot = original_cot[:error_index]
                            prompt_used = truncated_cot + intervention_phrase
                            
                            # Check if this intervention prompt is already cached
                            cached_solution, _ = _get_from_cache(prompt_used)
                            if cached_solution is None:
                                cache_key = _get_cache_key(prompt_used)
                                cache = _load_generation_cache()
                                cache[cache_key] = {
                                    'problem_text': prompt_used,
                                    'full_solution': intervened_cot,
                                    'extracted_answer': intervened_answer,
                                    'timestamp': f"intervention {condition} from {excel_file}",
                                    'model_id': MODEL_ID
                                }
                                intervention_added += 1
                
            total_baseline_added += baseline_added
            total_intervention_added += intervention_added
            # Suppressed individual file processing messages for cleaner output
                
        except Exception as e:
            print_timestamped_message(f"Could not process {excel_file}: {e}")
    
    total_added = total_baseline_added + total_intervention_added
    if total_added > 0:
        # Save the updated cache to file
        try:
            with open(GENERATION_CACHE_FILE, 'w', encoding='utf-8', newline='') as f:
                fieldnames = ['cache_key', 'problem_text', 'full_solution', 'extracted_answer', 'timestamp', 'model_id']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                cache = _load_generation_cache()
                for cache_key, entry in cache.items():
                    writer.writerow({
                        'cache_key': cache_key,
                        'problem_text': entry['problem_text'],
                        'full_solution': entry['full_solution'],
                        'extracted_answer': entry['extracted_answer'],
                        'timestamp': entry['timestamp'],
                        'model_id': entry['model_id']
                    })
            print_timestamped_message(f"Pre-populated cache with {total_baseline_added} baseline + {total_intervention_added} intervention = {total_added} total entries")
        except Exception as e:
            print_timestamped_message(f"Could not save pre-populated cache: {e}")
    else:
        print_timestamped_message("No new entries to add to cache from Excel files")

def save_to_excel(df, sheet_name, excel_path):
    """
    Saves a DataFrame to a specific sheet in an Excel file, creating or appending.
    """
    # Save data to Excel (suppressed verbose output for cleaner logs)
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        # If the file doesn't exist, create it
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print_timestamped_message(f"Created new file '{excel_path}' and saved to '{sheet_name}'.")
    except Exception as e:
        print_timestamped_message(f"ERROR: Failed to save to Excel. Reason: {e}")


# --- Model Loading ---
# We can also handle model loading here to keep the notebook clean.


# --- Text Generation ---
def _reconstruct_prompt(problem_text):
    """
    Reconstructs the prompt that would be sent to the model for a given problem.
    This is used to provide the raw input even for cached results.
    """
    system_prompt = (
        "You are a helpful assistant that solves math problems step-by-step. "
        "Your goal is to solve the user's problem. Do not make up new problems. "
        "Figure out the solution to the problem step by step. Use <think> tags to enclose your steps. "
        "After your reasoning, you MUST provide the final answer in the format \\boxed{answer}."
    )
    
    examples = [
        {
            "problem": "Solve the quadratic equation $x^2 - 5x + 6 = 0$.",
            "solution": "<think>I need to solve this quadratic equation x² - 5x + 6 = 0.</think>\n\n<think>Let me try factoring. I need two numbers that multiply to give 6 and add to give -5.</think>\n\nI'm looking for two numbers that multiply to 6 and add to -5.\n\n<think>Let me think... 6 = 1×6 = 2×3 = (-1)×(-6) = (-2)×(-3).</think>\n\n<think>For the sum: 1+6=7, 2+3=5, (-1)+(-6)=-7, (-2)+(-3)=-5. Yes! -2 and -3 work.</think>\n\nSo I can factor as (x - 2)(x - 3) = 0.\n\n<think>Now I use the zero product property: if AB = 0, then A = 0 or B = 0.</think>\n\n<think>So either x - 2 = 0 or x - 3 = 0.</think>\n\n<think>From x - 2 = 0, I get x = 2. From x - 3 = 0, I get x = 3.</think>\n\nLet me verify: <think>For x = 2: (2)² - 5(2) + 6 = 4 - 10 + 6 = 0 ✓</think> <think>For x = 3: (3)² - 5(3) + 6 = 9 - 15 + 6 = 0 ✓</think>",
            "answer": "x = 2 \\text{ or } x = 3"
        },
        {
            "problem": "Find the derivative of $f(x) = x^3 + 2x^2 - 5x + 1$.",
            "solution": "<think>I need to find the derivative of f(x) = x³ + 2x² - 5x + 1.</think>\n\n<think>I'll use the power rule: d/dx[xⁿ] = nxⁿ⁻¹.</think>\n\nLet me take the derivative of each term:\n\n<think>For x³: the power rule gives 3x³⁻¹ = 3x².</think>\n- The derivative of x³ is 3x²\n\n<think>For 2x²: the coefficient stays, power rule gives 2 × 2x²⁻¹ = 4x.</think>\n- The derivative of 2x² is 4x\n\n<think>For -5x: this is -5x¹, so -5 × 1x¹⁻¹ = -5x⁰ = -5.</think>\n- The derivative of -5x is -5\n\n<think>For the constant 1: derivatives of constants are always 0.</think>\n- The derivative of the constant 1 is 0\n\n<think>Now I combine all the terms: 3x² + 4x + (-5) + 0 = 3x² + 4x - 5.</think>",
            "answer": "f'(x) = 3x^2 + 4x - 5"
        },
        {
            "problem": "If $\\log_2(x) + \\log_2(x-3) = 2$, find $x$.",
            "solution": "<think>I have log₂(x) + log₂(x-3) = 2. This involves logarithms.</think>\n\n<think>I can use the logarithm property: log_a(m) + log_a(n) = log_a(mn).</think>\n\nSo log₂(x) + log₂(x-3) = log₂(x(x-3)).\n\n<think>Let me expand: x(x-3) = x² - 3x.</think>\n\nSo the equation becomes log₂(x² - 3x) = 2.\n\n<think>To convert from logarithmic to exponential form: if log_a(y) = b, then y = a^b.</think>\n\n<think>So x² - 3x = 2² = 4.</think>\n\nThis gives me x² - 3x = 4, or x² - 3x - 4 = 0.\n\n<think>Now I need to factor x² - 3x - 4. I need two numbers that multiply to -4 and add to -3.</think>\n\n<think>Let me try: -4 × 1 = -4 and -4 + 1 = -3. Perfect!</think>\n\nSo (x - 4)(x + 1) = 0.\n\n<think>This means x - 4 = 0 or x + 1 = 0.</think>\n\n<think>So x = 4 or x = -1.</think>\n\n<think>But wait, I need to check domain restrictions for the original logarithms.</think>\n\n<think>For log₂(x), I need x > 0.</think>\n\n<think>For log₂(x-3), I need x - 3 > 0, which means x > 3.</think>\n\n<think>So I need x > 3. This eliminates x = -1 since -1 < 3.</think>\n\n<think>Let me check x = 4: 4 > 3 ✓, so x = 4 is valid.</think>",
            "answer": "x = 4"
        }
    ]
    
    # Build examples section
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Problem: {ex['problem']}\n"
        examples_text += f"Solution:\n{ex['solution']}\n"
        examples_text += f"The final answer is \\boxed{{{ex['answer']}}}.\n\n"
    
    # DeepSeek-R1 specific prompt: Let model naturally start with <think>
    prompt = (
        f"Solve this math problem step by step. Show your reasoning and provide the final answer in \\boxed{{}} format.\n\n"
        f"Problem: {problem_text}\n\n"
    )
    
    return prompt

def generate_solution(model, tokenizer, problem_text, logits_processors=None, return_prompt=False):
    """
    Generates a step-by-step solution for a given math problem.
    Uses a one-shot example and a system prompt for robust format control.
    Supports caching to avoid regenerating identical problems.
    
    Args:
        return_prompt: If True, returns (solution, prompt) tuple instead of just solution
    """
    # Check cache first
    cached_solution, cached_answer = _get_from_cache(problem_text)
    if cached_solution is not None:
        if return_prompt:
            # We need to reconstruct the prompt for cached results
            prompt = _reconstruct_prompt(problem_text)
            return cached_solution, prompt
        return cached_solution
    
    # A more robust prompt structure with a system message and a concise example.
    system_prompt = (
        "You are a helpful assistant that solves math problems step-by-step. "
        "Your goal is to solve the user's problem. Do not make up new problems. "
        "Figure out the solution to the problem step by step. Use <think> tags to enclose your steps. "
        "After your reasoning, you MUST provide the final answer in the format \\boxed{answer}."
    )
    
    examples = [
        {
            "problem": "Solve the quadratic equation $x^2 - 5x + 6 = 0$.",
            "solution": "<think>I need to solve this quadratic equation x² - 5x + 6 = 0.</think>\n\n<think>Let me try factoring. I need two numbers that multiply to give 6 and add to give -5.</think>\n\nI'm looking for two numbers that multiply to 6 and add to -5.\n\n<think>Let me think... 6 = 1×6 = 2×3 = (-1)×(-6) = (-2)×(-3).</think>\n\n<think>For the sum: 1+6=7, 2+3=5, (-1)+(-6)=-7, (-2)+(-3)=-5. Yes! -2 and -3 work.</think>\n\nSo I can factor as (x - 2)(x - 3) = 0.\n\n<think>Now I use the zero product property: if AB = 0, then A = 0 or B = 0.</think>\n\n<think>So either x - 2 = 0 or x - 3 = 0.</think>\n\n<think>From x - 2 = 0, I get x = 2. From x - 3 = 0, I get x = 3.</think>\n\nLet me verify: <think>For x = 2: (2)² - 5(2) + 6 = 4 - 10 + 6 = 0 ✓</think> <think>For x = 3: (3)² - 5(3) + 6 = 9 - 15 + 6 = 0 ✓</think>",
            "answer": "x = 2 \\text{ or } x = 3"
        },
        {
            "problem": "Find the derivative of $f(x) = x^3 + 2x^2 - 5x + 1$.",
            "solution": "<think>I need to find the derivative of f(x) = x³ + 2x² - 5x + 1.</think>\n\n<think>I'll use the power rule: d/dx[xⁿ] = nxⁿ⁻¹.</think>\n\nLet me take the derivative of each term:\n\n<think>For x³: the power rule gives 3x³⁻¹ = 3x².</think>\n- The derivative of x³ is 3x²\n\n<think>For 2x²: the coefficient stays, power rule gives 2 × 2x²⁻¹ = 4x.</think>\n- The derivative of 2x² is 4x\n\n<think>For -5x: this is -5x¹, so -5 × 1x¹⁻¹ = -5x⁰ = -5.</think>\n- The derivative of -5x is -5\n\n<think>For the constant 1: derivatives of constants are always 0.</think>\n- The derivative of the constant 1 is 0\n\n<think>Now I combine all the terms: 3x² + 4x + (-5) + 0 = 3x² + 4x - 5.</think>",
            "answer": "f'(x) = 3x^2 + 4x - 5"
        },
        {
            "problem": "If $\\log_2(x) + \\log_2(x-3) = 2$, find $x$.",
            "solution": "<think>I have log₂(x) + log₂(x-3) = 2. This involves logarithms.</think>\n\n<think>I can use the logarithm property: log_a(m) + log_a(n) = log_a(mn).</think>\n\nSo log₂(x) + log₂(x-3) = log₂(x(x-3)).\n\n<think>Let me expand: x(x-3) = x² - 3x.</think>\n\nSo the equation becomes log₂(x² - 3x) = 2.\n\n<think>To convert from logarithmic to exponential form: if log_a(y) = b, then y = a^b.</think>\n\n<think>So x² - 3x = 2² = 4.</think>\n\nThis gives me x² - 3x = 4, or x² - 3x - 4 = 0.\n\n<think>Now I need to factor x² - 3x - 4. I need two numbers that multiply to -4 and add to -3.</think>\n\n<think>Let me try: -4 × 1 = -4 and -4 + 1 = -3. Perfect!</think>\n\nSo (x - 4)(x + 1) = 0.\n\n<think>This means x - 4 = 0 or x + 1 = 0.</think>\n\n<think>So x = 4 or x = -1.</think>\n\n<think>But wait, I need to check domain restrictions for the original logarithms.</think>\n\n<think>For log₂(x), I need x > 0.</think>\n\n<think>For log₂(x-3), I need x - 3 > 0, which means x > 3.</think>\n\n<think>So I need x > 3. This eliminates x = -1 since -1 < 3.</think>\n\n<think>Let me check x = 4: 4 > 3 ✓, so x = 4 is valid.</think>",
            "answer": "x = 4"
        }
    ]
    
    # Build examples section
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Problem: {ex['problem']}\n"
        examples_text += f"Solution:\n{ex['solution']}\n"
        examples_text += f"The final answer is \\boxed{{{ex['answer']}}}.\n\n"
    
    # DeepSeek-R1 specific prompt: Let model naturally start with <think>
    # Based on research: Don't force <think> in prompt, let model generate it
    prompt = (
        f"Solve this math problem step by step. Show your reasoning and provide the final answer in \\boxed{{}} format.\n\n"
        f"Problem: {problem_text}\n\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # HuggingFace generation with typical arguments
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    solution_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Extract answer and save to cache
    extracted_answer = extract_boxed_answer(solution_text)
    _save_to_cache(problem_text, solution_text, extracted_answer)
    
    if return_prompt:
        return solution_text, prompt
    return solution_text


def save_experiment_metadata(excel_path, model_id, tokenizer):
    """
    Saves comprehensive experiment metadata for full reproducibility.
    """
    metadata = pd.DataFrame([{
        'timestamp': get_timestamp_in_rome(),
        'model_id': model_id,
        'device': DEVICE,
        'debug_mode': DEBUG_MODE,
        'debug_sample_size': SAMPLE_SIZE,
        'prod_sample_size': PROD_SAMPLE_SIZE,
        'generation_temperature': 0.8,
        'generation_top_p': 0.92,
        'generation_max_tokens': 2048,
        'generation_do_sample': True,
        'intervention_phrases': str({
            "Corrective_Strong": " Wait, I made a mistake.",
            **CURRENT_INTERVENTIONS
        }),
        'error_analysis_model': 'claude-3-5-sonnet-20240620',
        'anthropic_api_used': 'ANTHROPIC_API_KEY' in os.environ,
        'dataset_name': 'EleutherAI/hendrycks_math',
        'dataset_subset': 'algebra',
        'dataset_split': 'test'
    }])
    save_to_excel(metadata, "Experiment_Metadata", excel_path)

def run_baseline_experiment(dataset, model, tokenizer, excel_path):
    """
    Runs the baseline experiment by generating solutions for a given dataset.
    Separates cached and uncached problems for clearer progress indication.
    """
    # --- 1. Separate cached and uncached problems ---
    uncached_examples = []
    cached_results = []
    
    print_timestamped_message("Checking cache for baseline results...")
    
    # Use a simple loop for the fast cache check instead of a tqdm bar
    for example in dataset:
        problem = example['problem']
        cached_solution, _ = _get_from_cache(problem)
        
        if cached_solution is not None:
            ground_truth = example['solution']
            ground_truth_answer = extract_boxed_answer(ground_truth)
            generated_answer = extract_boxed_answer(cached_solution)
            correct = is_correct(generated_answer, ground_truth_answer)
            
            # Reconstruct prompt for cached results
            raw_prompt = _reconstruct_prompt(problem)
            
            cached_results.append({
                "problem": problem,
                "raw_prompt": raw_prompt,
                "ground_truth_full": ground_truth,
                "ground_truth_answer": ground_truth_answer,
                "full_generated_solution": cached_solution,
                "generated_answer": generated_answer,
                "is_correct": correct,
                "error": None
            })
        else:
            uncached_examples.append(example)

    cache_hits = len(cached_results)
    total_problems = len(dataset)
    print_timestamped_message(f"Baseline: Found {cache_hits}/{total_problems} cached results.")

    # --- 2. Process uncached problems with a progress bar ---
    processed_results = []
    if uncached_examples:
        pbar = tqdm(uncached_examples, desc="Baseline Generation", leave=False)
        for example in pbar:
            problem = example['problem']
            ground_truth = example['solution']
            # Define ground_truth_answer here, outside the try block
            ground_truth_answer = extract_boxed_answer(ground_truth)
            
            try:
                full_solution, raw_prompt = generate_solution(model, tokenizer, problem, return_prompt=True)
                generated_answer = extract_boxed_answer(full_solution)
                correct = is_correct(generated_answer, ground_truth_answer)
                error_message = None
            except Exception as e:
                print_timestamped_message(f"ERROR generating solution for problem: {problem[:80]}... Details: {e}")
                full_solution = f"Error generating solution: {str(e)}"
                raw_prompt = _reconstruct_prompt(problem)  # Still provide prompt for error cases
                generated_answer = None
                correct = False
                error_message = traceback.format_exc()
            
            processed_results.append({
                "problem": problem,
                "raw_prompt": raw_prompt,
                "ground_truth_full": ground_truth,
                "ground_truth_answer": ground_truth_answer,
                "full_generated_solution": full_solution,
                "generated_answer": generated_answer,
                "is_correct": correct,
                "error": error_message
            })
        pbar.close()

    # --- 3. Combine, save, and summarize ---
    final_results_list = cached_results + processed_results
    
    # Add problem_id for traceability
    for i, result in enumerate(final_results_list):
        result['problem_id'] = i
        
    results_df = pd.DataFrame(final_results_list)
    save_to_excel(results_df, "Baseline_Results", excel_path)
    
    # Print summary
    accuracy = (results_df['is_correct'].sum() / len(results_df)) * 100 if not results_df.empty else 0
    errors = results_df['error'].count()
    print_timestamped_message(f"Baseline complete. Accuracy: {accuracy:.2f}%. Generations with errors: {errors}.")
    
    return results_df


# --- Evaluation Logic ---
def extract_boxed_answer(text: str):
    """
    Extracts the content from the last \\boxed{...} block in a string.
    """
    matches = re.findall(r'\\boxed\{(.*?)\}', str(text))
    if matches:
        return matches[-1]
    return None

def is_correct(ground_truth_answer, generated_answer):
    """
    Compares two answers for equivalence, handling multiple numbers, and basic LaTeX.
    This is a much more robust version.
    """
    if ground_truth_answer is None or generated_answer is None:
        return False

    # Normalize by removing ALL whitespace and converting to lowercase
    gt_norm = str(ground_truth_answer).replace(" ", "").replace("\t", "").replace("\n", "").strip().lower()
    gen_norm = str(generated_answer).replace(" ", "").replace("\t", "").replace("\n", "").strip().lower()

    # Direct comparison for non-numerical answers
    if gt_norm == gen_norm:
        return True

    # Numerical comparison logic
    # This regex handles integers, decimals, complex numbers, and basic fractions like \frac{a}{b}
    num_regex = r"-?\\d*\\.?\\d+i?|\\frac{\s*-?\\d+\s*}{\s*-?\\d+\s*}"
    
    gt_nums = re.findall(num_regex, gt_norm)
    gen_nums = re.findall(num_regex, gen_norm)

    # If there's a different number of numerical parts, they can't be equal
    if len(gt_nums) != len(gen_nums):
        return False
    
    # If no numbers found after all, and strings didn't match, they are not equal
    if not gt_nums:
        return False

    # Compare all numerical parts
    # For now, we'll do a direct string comparison of the extracted parts.
    # This correctly handles "4, 6, 14, 15" vs "4, 6, 14, 19"
    return gt_nums == gen_nums


# --- Intervention Logic ---
NOWAIT_KEYWORDS = [
    "wait", "alternatively", "hmm", "but", "however", "alternative", "another",
    "check", "double-check", "oh", "maybe", "verify", "other", "again",
    "now", "ah", "any"
]

class NoWaitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, keywords):
        self.tokenizer = tokenizer
        self.keywords = keywords
        self.suppress_token_ids = self._get_suppress_token_ids()

    def _get_suppress_token_ids(self):
        suppress_ids = []
        vocab = self.tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            decoded_token = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if any(keyword.lower() in decoded_token.lower() for keyword in self.keywords):
                suppress_ids.append(token_id)
        
        print(f"Identified {len(suppress_ids)} tokens to suppress for NoWait intervention.")
        return suppress_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.suppress_token_ids] = -float("inf")
        return scores


def save_patching_setup(source_text, destination_text, source_answer, destination_answer, 
                        problem_id, corrective_condition, excel_path):
    """
    Saves activation patching setup details for full reproducibility.
    """
    setup_data = pd.DataFrame([{
        'timestamp': get_timestamp_in_rome(),
        'selected_problem_id': problem_id,
        'corrective_condition_used': corrective_condition,
        'source_prompt': source_text,
        'destination_prompt': destination_text,
        'source_answer_tokens': source_answer,
        'destination_answer_tokens': destination_answer,
        'methodology': 'Activation patching: corrected intervention vs original uninformed continuation',
        'comparison_type': 'clean_corrected_vs_corrupted_original',
        'patching_position': 'last_token',
        'components_tested': 'all_attention_heads_and_mlp_layers'
    }])
    save_to_excel(setup_data, "Patching_Setup", excel_path)

def get_patching_results(
    model, 
    source_text, 
    destination_text,
    source_answer,
    destination_answer,
    excel_path,  # Add excel_path parameter
    problem_id=None,
    corrective_condition=None
):
    """
    Performs activation patching between a clean and corrupted run.
    Measures the change in logit difference for the correct vs. incorrect next token.
    """
    # Save patching setup for reproducibility
    save_patching_setup(source_text, destination_text, source_answer, destination_answer,
                       problem_id, corrective_condition, excel_path)
    
    # 1. Tokenize all inputs
    source_tokens = model.to_tokens(source_text)
    dest_tokens = model.to_tokens(destination_text)

    # Ensure tokenizer has a padding token ID
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        
    # For patching, prompts must have the same length. We pad the shorter one on the left.
    len_source = source_tokens.shape[1]
    len_dest = dest_tokens.shape[1]

    if len_source > len_dest:
        padding_needed = len_source - len_dest
        # Create padding tensor on the correct device
        padding = torch.full((1, padding_needed), model.tokenizer.pad_token_id, device=dest_tokens.device)
        dest_tokens = torch.cat([padding, dest_tokens], dim=1)
        print(f"Padded destination prompt with {padding_needed} tokens.")
    elif len_dest > len_source:
        padding_needed = len_dest - len_source
        # Create padding tensor on the correct device
        padding = torch.full((1, padding_needed), model.tokenizer.pad_token_id, device=source_tokens.device)
        source_tokens = torch.cat([padding, source_tokens], dim=1)
        print(f"Padded source prompt with {padding_needed} tokens.")

    # The "answer" tokens are the first token of the correct/incorrect continuation
    source_ans_token = model.to_tokens(source_answer, prepend_bos=False)[0, 0]
    dest_ans_token = model.to_tokens(destination_answer, prepend_bos=False)[0, 0]

    # For patching, prompts must have the same length
    assert source_tokens.shape[1] == dest_tokens.shape[1]
    patching_pos = source_tokens.shape[1] - 1

    # 2. Run the clean source run and cache all necessary activations
    _, clean_cache = model.run_with_cache(source_tokens)
    
    # 3. Get the original logits from the corrupted run to establish a baseline
    corrupted_logits = model(dest_tokens)

    def get_logit_diff(logits, correct_token, incorrect_token):
        # We only care about the logits for the token immediately following the prompt
        last_token_logits = logits[0, patching_pos, :]
        return last_token_logits[correct_token] - last_token_logits[incorrect_token]

    original_logit_diff = get_logit_diff(corrupted_logits, source_ans_token, dest_ans_token)
    print(f"Original logit difference: {original_logit_diff.item():.4f}")

    results = []

    def patching_hook(activation, hook, head_idx=None):
        clean_activation = clean_cache[hook.name]
        # Patch at the specific token position. The slicing logic handles both
        # attention heads and MLP/residual stream layers.
        if head_idx is not None:
            activation[0, patching_pos, head_idx, :] = clean_activation[0, patching_pos, head_idx, :]
        else:
            activation[0, patching_pos, :] = clean_activation[0, patching_pos, :]
        return activation

    # 4. Iterate through components, patching and recording the effect
    total_layers = model.cfg.n_layers
    print_timestamped_message(f"🎯 Analyzing {total_layers} attention layers...")
    
    # Attention Heads
    for layer in tqdm(range(model.cfg.n_layers), desc="Attention Patching", leave=False):
        for head in range(model.cfg.n_heads):
            hook_name = get_act_name('z', layer)
            hook_fn = lambda act, hook: patching_hook(act, hook, head_idx=head)
            patched_logits = model.run_with_hooks(dest_tokens, fwd_hooks=[(hook_name, hook_fn)])
            patched_logit_diff = get_logit_diff(patched_logits, source_ans_token, dest_ans_token)
            results.append({
                "layer": layer, 
                "head": head,
                "component": f"Head {head}",
                "logit_diff_change": (patched_logit_diff - original_logit_diff).item(),
                "original_logit_diff": original_logit_diff.item(),
                "patched_logit_diff": patched_logit_diff.item(),
                "component_type": "attention"
            })
            
    # MLP Layers
    print_timestamped_message(f"🎯 Analyzing {total_layers} MLP layers...")
    for layer in tqdm(range(model.cfg.n_layers), desc="MLP Patching", leave=False):
        hook_name = get_act_name('mlp_out', layer)
        hook_fn = lambda act, hook: patching_hook(act, hook)
        patched_logits = model.run_with_hooks(dest_tokens, fwd_hooks=[(hook_name, hook_fn)])
        patched_logit_diff = get_logit_diff(patched_logits, source_ans_token, dest_ans_token)
        results.append({
            "layer": layer, 
            "head": None,
            "component": "MLP",
            "logit_diff_change": (patched_logit_diff - original_logit_diff).item(),
            "original_logit_diff": original_logit_diff.item(),
            "patched_logit_diff": patched_logit_diff.item(),
            "component_type": "mlp"
        })
        
    results_df = pd.DataFrame(results)
    save_to_excel(results_df, "Activation_Patching", excel_path)
    print_timestamped_message("Activation patching complete.")
    
    return results_df


# --- Error Analysis with External LLM ---
import anthropic
import os
import time

def find_error_in_cot_with_think_blocks(problem, incorrect_cot):
    """
    Uses Claude Sonnet to identify which specific <think> block contains the first mistake.
    Returns the mistake block content for intervention targeting.
    """
    # FIRST: Check cache
    cache_key = _get_error_cache_key(problem, incorrect_cot + "_think_blocks")
    cached_result = _get_from_error_cache_by_key(cache_key)
    if cached_result is not None:
        return (cached_result, "cached", "cached")
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        error_msg = "ANTHROPIC_API_KEY not set - cannot identify errors"
        _save_to_error_cache_by_key(cache_key, error_msg)
        return (error_msg, "no_api_key", "no_api_key")
    
    try:
        # Extract and number the <think> blocks
        import re
        think_blocks = re.findall(r'<think>(.*?)</think>', incorrect_cot, re.DOTALL)
        
        if not think_blocks:
            result = "No <think> blocks found in solution"
            _save_to_error_cache_by_key(cache_key, result)
            return (result, "no_think_blocks", "no_think_blocks")
        
        # Create numbered list of think blocks
        numbered_blocks = []
        for i, block in enumerate(think_blocks, 1):
            clean_block = block.strip()
            numbered_blocks.append(f"Block {i}: {clean_block}")
        
        blocks_display = "\\n\\n".join(numbered_blocks)
        
        prompt = f"""You are analyzing a mathematical solution with multiple thinking steps. The solution contains <think> blocks numbered below.

ORIGINAL PROBLEM:
{problem}

NUMBERED THINK BLOCKS:
{blocks_display}

TASK: Identify the FIRST <think> block that contains a mathematical mistake.

Look for:
- Wrong arithmetic calculations
- Incorrect formulas or algebraic manipulations  
- Logical errors in reasoning steps
- Sign errors or computational mistakes

CRITICAL: Respond with ONLY the block number (like "2" or "3").
If no mathematical mistake exists, or if no thinking blocks are found, respond with "0".
Do NOT explain or add any other text."""

        # Call Anthropic API with rate limiting
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,  # Very small since we only expect a number
                messages=[{"role": "user", "content": prompt}]
            )
            time.sleep(2)
        except Exception as api_error:
            if 'rate_limit_error' in str(api_error):
                print_timestamped_message("Rate limit hit in think-block detection. Waiting 30 seconds...")
                time.sleep(30)
                try:
                    message = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=10,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    time.sleep(3)
                except Exception as retry_error:
                    error_msg = f"Rate limit retry failed: {str(retry_error)}"
                    _save_to_error_cache_by_key(cache_key, error_msg)
                    return (error_msg, prompt, str(retry_error))
            else:
                error_msg = f"API error in think-block detection: {str(api_error)}"
                _save_to_error_cache_by_key(cache_key, error_msg)
                return (error_msg, prompt, str(api_error))
        
        response_text = message.content[0].text.strip()
        
        # Validate and process response
        try:
            block_number = int(response_text)
            
            if block_number == 0:
                result = "No mistake found by Anthropic"
            elif 1 <= block_number <= len(think_blocks):
                # Return the problematic think block content
                mistake_block = think_blocks[block_number - 1].strip()
                result = f"<think>{mistake_block}</think>"
            else:
                result = f"Invalid block number {block_number} (available: 1-{len(think_blocks)})"
                
        except ValueError:
            result = f"Could not parse block number from: {response_text}"
        
        # Cache and return result
        _save_to_error_cache_by_key(cache_key, result)
        return (result, prompt, response_text)
        
    except Exception as e:
        error_msg = f"Error in think-block analysis: {str(e)}"
        _save_to_error_cache_by_key(cache_key, error_msg)
        return (error_msg, prompt if 'prompt' in locals() else "unknown", str(e))

def find_error_in_cot_with_tokens(problem, incorrect_cot, tokenizer):
    """
    Uses token indexing with Claude Sonnet to identify the first point of mistake in a CoT.
    Returns tuple: (mistake_sentence, raw_input, raw_output) for debugging.
    """
    # FIRST: Check if we have a legacy cache entry (prioritize existing cache)
    legacy_cached = _get_from_error_cache(problem, incorrect_cot)
    if legacy_cached is not None:
        return (legacy_cached, "cached", "cached")
    
    # SECOND: Check token-based cache
    cache_key = _get_error_cache_key(problem, incorrect_cot + "_tokenized")
    cached_result = _get_from_error_cache_by_key(cache_key)
    if cached_result is not None:
        return (cached_result, "cached", "cached")
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        error_msg = "ANTHROPIC_API_KEY not set - cannot identify errors"
        _save_to_error_cache_by_key(cache_key, error_msg)
        return (error_msg, "no_api_key", "no_api_key")
    
    try:
        # Tokenize the CoT and create numbered list
        tokens = tokenizer.encode(incorrect_cot)
        token_texts = [tokenizer.decode([token]) for token in tokens]
        
        # Create numbered token display (limit to reasonable length)
        max_tokens = 800  # Conservative limit for API context (leave room for prompt)
        if len(token_texts) > max_tokens:
            token_texts = token_texts[:max_tokens]
            truncated_note = f"\\n[Truncated to first {max_tokens} tokens]"
        else:
            truncated_note = ""
        
        # Safety check for very short inputs
        if len(token_texts) < 5:
            return "CoT too short for token-based analysis"
        
        numbered_tokens = []
        for i, token_text in enumerate(token_texts):
            # Clean token display and handle special characters
            clean_token = repr(token_text).strip("'\"")
            if len(clean_token) > 50:  # Limit very long tokens
                clean_token = clean_token[:47] + "..."
            numbered_tokens.append(f"{i:3d}: {clean_token}")
        
        tokens_display = "\\n".join(numbered_tokens) + truncated_note
        
        prompt = f"""You are analyzing a mathematical solution that has been tokenized. Each line shows: "INDEX: TOKEN_TEXT"

ORIGINAL PROBLEM:
{problem}

TOKENIZED SOLUTION:
{tokens_display}

TASK: Find the FIRST token where a mathematical mistake occurs. Look for:
- Wrong arithmetic (e.g., 2+2=5)
- Incorrect formulas or substitutions  
- Logical errors in steps
- Sign errors or algebraic mistakes

CRITICAL: Only respond with the integer index number (like "42"). 
If you cannot find any mathematical mistake, respond with "-1".
Do NOT explain or add any other text."""

        # Call Anthropic API with rate limiting
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,  # Very small since we only expect an integer
                messages=[{"role": "user", "content": prompt}]
            )
            # Add delay to respect rate limits
            time.sleep(2)
        except Exception as api_error:
            if 'rate_limit_error' in str(api_error):
                print_timestamped_message("Rate limit hit in token-based detection. Waiting 30 seconds...")
                time.sleep(30)
                try:
                    message = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=50,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    time.sleep(3)  # Longer delay after recovery
                except Exception as retry_error:
                    error_msg = f"Rate limit retry failed: {str(retry_error)}"
                    _save_to_error_cache_by_key(cache_key, error_msg)
                    return (error_msg, prompt, str(retry_error))
            else:
                error_msg = f"API error in token detection: {str(api_error)}"
                _save_to_error_cache_by_key(cache_key, error_msg)
                return (error_msg, prompt, str(api_error))
        
        response_text = message.content[0].text.strip()
        
        # Validate response quality - check for hallucinations
        if len(response_text) > 20:
            result = f"Response too long, possible hallucination: {response_text[:50]}..."
        elif any(word in response_text.lower() for word in ['provided', 'text', 'appears', 'incomplete', 'findable']):
            result = f"Anthropic hallucination detected: {response_text}"
        else:
            # Parse the response to extract token index
            try:
                error_token_index = int(response_text)
                
                if error_token_index == -1:
                    result = "No mistake found by Anthropic"
                elif 0 <= error_token_index < len(token_texts):
                    # Convert token index to character index in original text
                    char_index = len(tokenizer.decode(tokens[:error_token_index]))
                    
                    # Extract mistake context (a reasonable sentence around the mistake)
                    mistake_start = max(0, char_index - 50)
                    mistake_end = min(len(incorrect_cot), char_index + 100)
                    mistake_context = incorrect_cot[mistake_start:mistake_end].strip()
                    
                    # Extract a good mistake sentence - try multiple delimiters
                    sentences = []
                    for delimiter in ['. ', '\\n', ';']:
                        potential_sentences = mistake_context.split(delimiter)
                        if len(potential_sentences) > 1:
                            sentences = potential_sentences
                            break
                    
                    if sentences and len(sentences) > 1:
                        # Find the sentence containing the mistake point
                        cumulative_len = mistake_start
                        for sentence in sentences:
                            sentence_end = cumulative_len + len(sentence) + len(delimiter)
                            if sentence_end > char_index and len(sentence.strip()) > 10:
                                result = sentence.strip()
                                break
                            cumulative_len = sentence_end
                        else:
                            result = mistake_context.strip()
                    else:
                        result = mistake_context.strip()
                    
                    # Ensure we have a reasonable mistake sentence (not too short or too long)
                    if len(result) < 15:
                        result = mistake_context.strip()
                    elif len(result) > 200:
                        result = result[:197] + "..."
                    
                else:
                    result = f"Invalid token index {error_token_index} (max: {len(token_texts)-1})"
                    
            except ValueError:
                # Fallback: try to extract any number from response
                import re
                numbers = re.findall(r'\\d+', response_text)
                if numbers:
                    error_token_index = int(numbers[0])
                    if 0 <= error_token_index < len(token_texts):
                        char_index = len(tokenizer.decode(tokens[:error_token_index]))
                        mistake_start = max(0, char_index - 50)
                        mistake_end = min(len(incorrect_cot), char_index + 100)
                        result = incorrect_cot[mistake_start:mistake_end].strip()
                    else:
                        result = f"Parsed invalid token index {error_token_index} from: {response_text}"
                else:
                    result = f"Could not parse token index from: {response_text}"
        
        # Cache and return result
        _save_to_error_cache_by_key(cache_key, result)
        return (result, prompt, response_text)
        
    except Exception as e:
        error_msg = f"Error calling Anthropic API: {str(e)}"
        _save_to_error_cache_by_key(cache_key, error_msg)
        return (error_msg, prompt if 'prompt' in locals() else "unknown", str(e))

def find_error_in_cot(problem, incorrect_cot):
    """
    Legacy function - uses string-based mistake detection.
    Supports caching to avoid re-analyzing identical problems.
    Returns tuple: (mistake_sentence, raw_input, raw_output) for debugging.
    """
    # Check cache first
    cached_error = _get_from_error_cache(problem, incorrect_cot)
    if cached_error is not None:
        return (cached_error, "cached", "cached")
    
    # Placeholder for the actual API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print_timestamped_message("Warning: ANTHROPIC_API_KEY not set. Returning a placeholder error.")
        # Return the last sentence as a fallback for testing without an API key
        fallback_error = incorrect_cot.strip().split('.')[-2] + '.'
        # Cache the fallback result too
        _save_to_error_cache(problem, incorrect_cot, fallback_error, "fallback_no_api_key")
        return (fallback_error, "no_api_key", "no_api_key")

    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""
        You are a meticulous math expert. Your task is to analyze a math problem and an incorrect solution provided in a Chain-of-Thought format.
        You must identify the single, first sentence where a logical or mathematical error occurs.

        Here is the problem:
        <problem>
        {problem}
        </problem>

        Here is the incorrect Chain-of-Thought solution:
        <cot>
        {incorrect_cot}
        </cot>

        Analyze the CoT carefully. Identify the exact, complete sentence where the very first mistake is made.
        Respond with ONLY that single sentence and nothing else.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # The response content will be a list of blocks, we want the text from the first one.
        error_sentence = response.content[0].text
        
        # Cache the successful result
        _save_to_error_cache(problem, incorrect_cot, error_sentence, "claude-3-5-sonnet-20240620")
        
        # Add delay to respect rate limits
        time.sleep(2)  # Increased from 1 second
        
        return (error_sentence, prompt, error_sentence)
    except Exception as e:
        print_timestamped_message(f"An error occurred with the Anthropic API: {e}")
        # Implement exponential backoff for rate limits
        if 'rate_limit_error' in str(e):
            print_timestamped_message("Rate limit hit. Waiting 30 seconds before retrying...")
            time.sleep(30)
            # Try one more time after rate limit
            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                error_sentence = response.content[0].text
                _save_to_error_cache(problem, incorrect_cot, error_sentence, "claude-3-5-sonnet-20240620")
                time.sleep(3)  # Longer delay after recovery
                return (error_sentence, prompt, error_sentence)
            except Exception as e2:
                print_timestamped_message(f"Retry failed: {e2}")
                return (f"Rate limit error, retry failed: {str(e2)}", prompt, str(e2))
        else:
            time.sleep(5)  # General API error recovery
        return (f"API error: {str(e)}", prompt if 'prompt' in locals() else "unknown", str(e))

def identify_errors(baseline_df, excel_path, tokenizer=None, use_token_indexing=True):
    """
    Identifies the first mistake sentence in incorrect CoTs using an external LLM.
    Separates cached and uncached problems for clearer progress indication.
    """
    incorrect_df = baseline_df[baseline_df['is_correct'] == False].copy()
    
    if incorrect_df.empty:
        print_timestamped_message("No incorrect answers to analyze. Skipping mistake identification.")
        save_to_excel(pd.DataFrame(), "Error_Analysis", excel_path)
        return pd.DataFrame()

    # --- 1. Separate cached and uncached problems ---
    uncached_rows = []
    cached_results = []
    
    print_timestamped_message("Checking cache for mistake detection results...")
    for _, row in incorrect_df.iterrows():
        problem = row['problem']
        cot = row['full_generated_solution']
        
        # Determine cache key based on method
        if use_token_indexing and tokenizer is not None:
            cache_key = _get_error_cache_key(problem, cot + "_think_blocks")
            cached_result = _get_from_error_cache_by_key(cache_key)
        else:
            cached_result = _get_from_error_cache(problem, cot)
            
        if cached_result is not None:
            result_row = row.to_dict()
            result_row['mistake_sentence'] = cached_result
            result_row['mistake_identification_error'] = None
            result_row['anthropic_raw_input'] = "cached"
            result_row['anthropic_raw_output'] = "cached"
            cached_results.append(result_row)
        else:
            uncached_rows.append(row)
            
    cache_hits = len(cached_results)
    total_incorrect = len(incorrect_df)
    print_timestamped_message(f"Mistake ID: Found {cache_hits}/{total_incorrect} cached results.")

    # --- 2. Process uncached problems ---
    processed_results = []
    if uncached_rows:
        # Choose mistake detection method
        if use_token_indexing and tokenizer is not None:
            print_timestamped_message("Using think-block approach for mistake detection...")
            mistake_function = lambda p, c: find_error_in_cot_with_think_blocks(p, c)
        else:
            print_timestamped_message("Using legacy string-based approach for mistake detection...")
            mistake_function = find_error_in_cot

        pbar = tqdm(uncached_rows, desc="Mistake Detection", leave=False)
        for row in pbar:
            result_row = row.to_dict()
            try:
                mistake_sentence, raw_input, raw_output = mistake_function(row['problem'], row['full_generated_solution'])
                result_row['mistake_sentence'] = mistake_sentence
                result_row['mistake_identification_error'] = None
                result_row['anthropic_raw_input'] = raw_input
                result_row['anthropic_raw_output'] = raw_output
            except Exception as e:
                print_timestamped_message(f"ERROR identifying mistake for problem: {row['problem'][:80]}... Details: {e}")
                result_row['mistake_sentence'] = None
                result_row['mistake_identification_error'] = traceback.format_exc()
                result_row['anthropic_raw_input'] = "exception_occurred"
            processed_results.append(result_row)
        pbar.close()

    # --- 3. Combine, save, and summarize ---
    final_results_list = cached_results + processed_results
    results_df = pd.DataFrame(final_results_list)

    if not results_df.empty:
        # CRITICAL: Add mistake sentence usability tracking
        results_df['mistake_sentence_usable'] = results_df.apply(
            lambda row: (pd.notna(row['mistake_sentence']) and 
                        pd.notna(row['full_generated_solution']) and
                        row['mistake_sentence'] in row['full_generated_solution']), axis=1
        )
        
        # Mark unusable mistake sentences as labeling failures
        results_df.loc[~results_df['mistake_sentence_usable'], 'mistake_identification_error'] = 'Mistake sentence not findable in CoT'

        # Print corrected summary
        attempted_count = len(results_df)
        usable_count = results_df['mistake_sentence_usable'].sum()
        total_failures = results_df['mistake_identification_error'].notna().sum()
        print_timestamped_message(f"Mistake identification complete. Successfully labeled {usable_count}/{attempted_count} problems ({usable_count/attempted_count*100:.1f}% if attempted > 0 else 0.0). Failures: {total_failures}.")
    
    save_to_excel(results_df, "Error_Analysis", excel_path)
    return results_df


def run_insertion_test(error_analysis_df, model, tokenizer, intervention_phrases, excel_path):
    """
    Runs the insertion test by injecting phrases at the point of error.
    Separates cached and uncached problems for clearer progress indication.
    """
    # --- 1. Filter for analyzable problems ---
    if 'mistake_sentence_usable' in error_analysis_df.columns:
        analyzable_df = error_analysis_df[error_analysis_df['mistake_sentence_usable'] == True].copy()
    else:
        analyzable_df = error_analysis_df.dropna(subset=['mistake_sentence', 'full_generated_solution']).copy()
    
    if analyzable_df.empty:
        print_timestamped_message("No analyzable mistakes found. Skipping insertion test.")
        save_to_excel(pd.DataFrame(), "Insertion_Test", excel_path)
        return pd.DataFrame()

    # --- 2. Separate cached and uncached interventions ---
    uncached_interventions = []
    cached_results = []

    print_timestamped_message("Checking cache for intervention results...")
    
    for _, row in analyzable_df.iterrows():
        original_cot = row['full_generated_solution']
        mistake_sentence = row['mistake_sentence']
        
        # Determine truncation point
        mistake_index = original_cot.find(mistake_sentence)
        if mistake_index == -1:
            continue # Should not happen due to pre-filtering
        truncated_cot = original_cot[:mistake_index]

        for condition, phrase in intervention_phrases.items():
            prompt_with_intervention = truncated_cot + phrase
            cached_solution, _ = _get_from_cache(prompt_with_intervention)
            
            if cached_solution is not None:
                ground_truth_answer = row['ground_truth_answer']
                intervened_answer = extract_boxed_answer(cached_solution)
                is_corrected = is_correct(intervened_answer, ground_truth_answer)
                
                cached_results.append({
                    'problem_id': row['problem_id'], 'problem': row['problem'], 'ground_truth_answer': ground_truth_answer,
                    'condition': condition, 'intervened_cot': cached_solution, 'intervened_answer': intervened_answer,
                    'is_corrected': is_corrected, 'error': None, 'mistake_sentence': mistake_sentence,
                    'truncation_point': mistake_index, 'intervention_phrase': phrase,
                    'original_cot_length': len(original_cot), 'truncated_cot_length': len(truncated_cot),
                    'final_prompt': prompt_with_intervention
                })
            else:
                uncached_interventions.append((row, condition, phrase, prompt_with_intervention, truncated_cot, mistake_index, original_cot))

    print_timestamped_message(f"Intervention Test: Found {len(cached_results)} cached results. Processing {len(uncached_interventions)} new generations.")

    # --- 3. Process uncached interventions ---
    processed_results = []
    if uncached_interventions:
        pbar_gen = tqdm(uncached_interventions, desc="Intervention Generation", leave=False)
        for row, condition, phrase, prompt, truncated_cot, mistake_index, original_cot in pbar_gen:
            try:
                intervened_cot = generate_solution(model, tokenizer, prompt)
                intervened_answer = extract_boxed_answer(intervened_cot)
                is_corrected = is_correct(intervened_answer, row['ground_truth_answer'])
                error_message = None
            except Exception as e:
                print_timestamped_message(f"ERROR during intervention for problem: {row['problem'][:80]}... Condition: {condition}. Details: {e}")
                intervened_cot = f"Error during generation: {str(e)}"
                intervened_answer = None
                is_corrected = False
                error_message = traceback.format_exc()

            processed_results.append({
                'problem_id': row['problem_id'], 'problem': row['problem'], 'ground_truth_answer': row['ground_truth_answer'],
                'condition': condition, 'intervened_cot': intervened_cot, 'intervened_answer': intervened_answer,
                'is_corrected': is_corrected, 'error': error_message, 'mistake_sentence': row['mistake_sentence'],
                'truncation_point': mistake_index, 'intervention_phrase': phrase,
                'original_cot_length': len(original_cot), 'truncated_cot_length': len(truncated_cot),
                'final_prompt': prompt
            })
        pbar_gen.close()

    # --- 4. Combine, save, and summarize ---
    final_results_list = cached_results + processed_results
    intervention_df = pd.DataFrame(final_results_list)
    save_to_excel(intervention_df, "Insertion_Test", excel_path)

    if not intervention_df.empty:
        tested_problems = len(set(intervention_df['problem_id']))
        print_timestamped_message(f"Insertion test complete: {tested_problems} problems with usable mistake labels tested")
        
        correction_rates = intervention_df.groupby('condition')['is_corrected'].mean() * 100
        print_timestamped_message("Insertion test complete. Correction rates:")
        print(correction_rates.to_string(float_format="%.2f%%"))

    return intervention_df
