
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
DEBUG_MODE = True
SAMPLE_SIZE = 10 # Number of examples to use in debug mode.
PROD_SAMPLE_SIZE = 150 # Limit the number of examples for the full run

# --- Global Variables & Constants ---
# It's good practice to define constants that might be used across functions.
# We can initialize them as None and set them from the notebook.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# --- Cache System ---
CACHE_DIR = "cache"
GENERATION_CACHE_FILE = os.path.join(CACHE_DIR, "generation_cache.csv")
ERROR_DETECTION_CACHE_FILE = os.path.join(CACHE_DIR, "error_detection_cache.csv")
_generation_cache = None  # In-memory cache
_error_detection_cache = None  # In-memory cache

# --- Utility Functions ---

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
        print_timestamped_message(f"Cache hit for problem: {problem_text[:50]}...")
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
        print_timestamped_message(f"Error detection cache hit for problem: {problem_text[:50]}...")
        return cached_result['error_sentence']
    
    return None

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
    
    # Define intervention phrases mapping
    intervention_phrases = {
        "Corrective_Strong": " Wait, I made a mistake.",
        "Corrective_Alert": " Hold on!",
        "Corrective_Reconsider": " Wait, I need to reconsider.",
        "Corrective_Wrong": " Wait, that was wrong!",
        "Corrective_Original": " Wait, let me re-evaluate that.",
        "Neutral": " And,",
        "Confirmation": " Continuing,"
    }
    
    print_timestamped_message(f"Pre-populating cache from {len(excel_files)} Excel files...")
    
    for excel_file in excel_files:
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
            print_timestamped_message(f"Processed {excel_file}: baseline={baseline_added}, interventions={intervention_added}")
                
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
    print_timestamped_message(f"Saving data to sheet '{sheet_name}' in '{excel_path}'...")
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print_timestamped_message(f"Successfully saved to '{sheet_name}'.")
    except FileNotFoundError:
        # If the file doesn't exist, create it
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print_timestamped_message(f"Created new file '{excel_path}' and saved to '{sheet_name}'.")
    except Exception as e:
        print_timestamped_message(f"ERROR: Failed to save to Excel. Reason: {e}")


# --- Model Loading ---
# We can also handle model loading here to keep the notebook clean.
def load_model_and_tokenizer():
    """Loads the pre-trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=None  # Avoid device splitting for TransformerLens compatibility
    )
    
    # Move model to the designated device
    model.to(DEVICE)
    
    # Fix for verbose warning: Set pad_token_id to eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

# --- Text Generation ---
def generate_solution(model, tokenizer, problem_text, logits_processors=None):
    """
    Generates a step-by-step solution for a given math problem.
    Uses a one-shot example and a system prompt for robust format control.
    Supports caching to avoid regenerating identical problems.
    """
    # Check cache first
    cached_solution, cached_answer = _get_from_cache(problem_text)
    if cached_solution is not None:
        return cached_solution
    
    # A more robust prompt structure with a system message and a concise example.
    system_prompt = (
        "You are a helpful assistant that solves math problems step-by-step. "
        "Your goal is to solve the user's problem. Do not make up new problems. "
        "After your reasoning, you MUST provide the final answer in the format \\boxed{answer}."
    )
    
    one_shot_example = (
        "Problem: What is the value of $x$ in the equation $2x + 3 = 11$?\\n"
        "Solution:\\n<think>To solve for x, I will subtract 3 from both sides, giving 2x = 8. Then I will divide by 2, giving x = 4.</think>\\n"
        "The final answer is \\boxed{4}."
    )
    
    # Construct the final prompt
    prompt = (
        f"{system_prompt}\\n\\n"
        f"Here is an example:\\n{one_shot_example}\\n\\n"
        f"---\\n\\n"
        f"Problem: {problem_text}\\n"
        f"Solution:\\n<think>"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.8,
        top_p=0.92,
        do_sample=True,
        logits_processor=logits_processors,
        pad_token_id=tokenizer.eos_token_id # Explicitly set pad_token_id
    )
    
    solution_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Extract answer and save to cache
    extracted_answer = extract_boxed_answer(solution_text)
    _save_to_cache(problem_text, solution_text, extracted_answer)
    
    return solution_text


def save_experiment_metadata(excel_path):
    """
    Saves comprehensive experiment metadata for full reproducibility.
    """
    metadata = pd.DataFrame([{
        'timestamp': get_timestamp_in_rome(),
        'model_id': MODEL_ID,
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
            "Corrective_Alert": " Hold on!",
            "Corrective_Reconsider": " Wait, I need to reconsider.",
            "Corrective_Wrong": " Wait, that was wrong!",
            "Corrective_Original": " Wait, let me re-evaluate that.",
            "Neutral": " And,",
            "Confirmation": " Continuing,"
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
    Runs the baseline experiment, generating solutions for each problem in the dataset.
    Saves results progressively to an Excel file.
    """
    # Save experiment metadata first
    save_experiment_metadata(excel_path)
    
    # Initialize cache (pre-population should be done once via setup_cache.py)
    print_timestamped_message("Loading generation cache...")
    cache = _load_generation_cache()
    print_timestamped_message(f"Cache loaded with {len(cache)} entries")
    
    results_list = []
    
    for i, example in enumerate(tqdm(dataset, desc="Running Baseline Experiment")):
        problem = example['problem']
        ground_truth = example['solution']
        
        try:
            full_solution = generate_solution(model, tokenizer, problem)
            generated_answer = extract_boxed_answer(full_solution)
            # CRITICAL FIX: Extract the ground truth answer before comparing
            ground_truth_answer = extract_boxed_answer(ground_truth)
            correct = is_correct(ground_truth_answer, generated_answer)
            error_message = None
        except Exception:
            full_solution = None
            generated_answer = None
            correct = False
            error_message = traceback.format_exc()
            
        results_list.append({
            "problem_id": i,
            "problem": problem,
            "ground_truth_full": ground_truth,
            "ground_truth_answer": ground_truth_answer,
            "full_generated_solution": full_solution,
            "generated_answer": generated_answer,
            "is_correct": correct,
            "error": error_message
        })
        
    results_df = pd.DataFrame(results_list)
    save_to_excel(results_df, "Baseline_Results", excel_path)
    
    # Print summary
    accuracy = (results_df['is_correct'].sum() / len(results_df)) * 100
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


def load_tl_model(hf_model):
    """
    Loads a pre-existing HuggingFace model into a HookedTransformer object.
    This is more robust for unsupported custom architectures.
    """
    print("Loading model into TransformerLens...")
    
    # First, ensure the HF model is on the correct device
    hf_model.to(DEVICE)
    
    try:
        # Try the standard method first
        tl_model = HookedTransformer.from_pretrained(
            MODEL_ID,
            hf_model=hf_model,
            torch_dtype=torch.bfloat16,
            device=DEVICE,
            trust_remote_code=True
        )
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"Device error encountered: {e}")
            print("Trying with from_pretrained_no_processing...")
            # Try the no-processing method for reduced precision models
            tl_model = HookedTransformer.from_pretrained_no_processing(
                MODEL_ID,
                hf_model=hf_model,
                torch_dtype=torch.bfloat16,
                device=DEVICE,
                trust_remote_code=True
            )
        else:
            raise e
    
    print("TransformerLens model loaded.")
    return tl_model

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
    # Attention Heads
    for layer in tqdm(range(model.cfg.n_layers), desc="Patching Attention Layers"):
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
    for layer in tqdm(range(model.cfg.n_layers), desc="Patching MLP Layers"):
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

def find_error_in_cot(problem, incorrect_cot):
    """
    Uses an external LLM (Claude Sonnet) to identify the first point of error in a CoT.
    NOTE: This function requires an ANTHROPIC_API_KEY environment variable to be set.
    Supports caching to avoid re-analyzing identical problems.
    """
    # Check cache first
    cached_error = _get_from_error_cache(problem, incorrect_cot)
    if cached_error is not None:
        return cached_error
    
    # Placeholder for the actual API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print_timestamped_message("Warning: ANTHROPIC_API_KEY not set. Returning a placeholder error.")
        # Return the last sentence as a fallback for testing without an API key
        fallback_error = incorrect_cot.strip().split('.')[-2] + '.'
        # Cache the fallback result too
        _save_to_error_cache(problem, incorrect_cot, fallback_error, "fallback_no_api_key")
        return fallback_error

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
        
        # Add a 1-second sleep to respect rate limits
        time.sleep(1)
        
        return error_sentence
    except Exception as e:
        print_timestamped_message(f"An error occurred with the Anthropic API: {e}")
        # Add a longer sleep if we hit a rate limit error to allow recovery
        if 'rate_limit_error' in str(e):
            time.sleep(10)
        return None

def identify_errors(baseline_df, excel_path):
    """
    Identifies the first error sentence in incorrect CoTs using an external LLM.
    """
    incorrect_df = baseline_df[baseline_df['is_correct'] == False].copy()
    
    if incorrect_df.empty:
        print_timestamped_message("No incorrect answers to analyze. Skipping error identification.")
        # Save an empty df to indicate the step was run
        save_to_excel(pd.DataFrame(), "Error_Analysis", excel_path)
        return pd.DataFrame()

    error_sentences = []
    errors = []

    for _, row in tqdm(incorrect_df.iterrows(), total=len(incorrect_df), desc="Finding Error Sentences"):
        try:
            error_sentence = find_error_in_cot(row['problem'], row['full_generated_solution'])
            error_sentences.append(error_sentence)
            errors.append(None)
        except Exception:
            error_sentences.append(None)
            errors.append(traceback.format_exc())

    incorrect_df['error_sentence'] = error_sentences
    incorrect_df['error_identification_error'] = errors
    
    save_to_excel(incorrect_df, "Error_Analysis", excel_path)
    
    # Print summary
    found_count = incorrect_df['error_sentence'].notna().sum()
    error_count = incorrect_df['error_identification_error'].notna().sum()
    print_timestamped_message(f"Error identification complete. Found sentences for {found_count}/{len(incorrect_df)} examples. Encountered {error_count} errors.")

    return incorrect_df


def run_insertion_test(error_analysis_df, model, tokenizer, intervention_phrases, excel_path):
    """
    Runs the insertion test by injecting phrases at the point of error.
    """
    analyzable_df = error_analysis_df.dropna(subset=['error_sentence', 'full_generated_solution']).copy()
    analyzable_df = analyzable_df[~analyzable_df['error_sentence'].str.contains("ANTHROPIC_API_KEY not set")]
    
    if analyzable_df.empty:
        print_timestamped_message("No analyzable errors found. Skipping insertion test.")
        save_to_excel(pd.DataFrame(), "Insertion_Test", excel_path)
        return pd.DataFrame()

    intervention_results = []

    for _, row in tqdm(analyzable_df.iterrows(), total=len(analyzable_df), desc="Running Insertion Test"):
        original_cot = row['full_generated_solution']
        error_sentence = row['error_sentence']
        
        # Find the start of the error sentence
        error_index = original_cot.find(error_sentence)
        if error_index == -1:
            continue # Skip if the sentence isn't found (should be rare)
            
        truncated_cot = original_cot[:error_index]

        for condition, phrase in intervention_phrases.items():
            prompt_with_intervention = truncated_cot + phrase
            
            try:
                intervened_cot = generate_solution(model, tokenizer, prompt_with_intervention)
                intervened_answer = extract_boxed_answer(intervened_cot)
                # CRITICAL FIX: Extract the ground truth answer from the original solution string
                gt_answer = extract_boxed_answer(row['ground_truth_full'])
                is_corrected = is_correct(gt_answer, intervened_answer)
                error_message = None
            except Exception:
                intervened_cot = None
                intervened_answer = None
                is_corrected = False
                error_message = traceback.format_exc()

            intervention_results.append({
                'problem_id': row['problem_id'],
                'problem': row['problem'],
                'condition': condition,
                'intervened_cot': intervened_cot,
                'intervened_answer': intervened_answer,
                'is_corrected': is_corrected,
                'error': error_message,
                # Add traceability fields
                'error_sentence': error_sentence,
                'truncation_point': error_index,
                'intervention_phrase': phrase,
                'original_cot_length': len(original_cot),
                'truncated_cot_length': len(truncated_cot)
            })

    intervention_df = pd.DataFrame(intervention_results)
    save_to_excel(intervention_df, "Insertion_Test", excel_path)

    # Print summary
    correction_rates = intervention_df.groupby('condition')['is_corrected'].mean() * 100
    print_timestamped_message("Insertion test complete. Correction rates:")
    print(correction_rates.to_string(float_format="%.2f%%"))

    return intervention_df
