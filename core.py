
# --- Activation Patching Logic ---
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
import re
from tqdm.auto import tqdm
import pandas as pd
from datetime import datetime
import pytz
import traceback

# This file will contain the core logic for our experiments, 
# keeping the main notebook clean and focused on the narrative.

# --- Experiment Configuration ---
# Set to True for a quick run on a small sample, False for the full overnight run.
DEBUG_MODE = False
SAMPLE_SIZE = 10 # Number of examples to use in debug mode.

# --- Global Variables & Constants ---
# It's good practice to define constants that might be used across functions.
# We can initialize them as None and set them from the notebook.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# --- Utility Functions ---

def get_timestamp_in_rome():
    """Returns the current timestamp in 'Europe/Rome' timezone."""
    rome_tz = pytz.timezone('Europe/Rome')
    return datetime.now(rome_tz).strftime('%Y-%m-%d %H:%M:%S %Z')

def print_timestamped_message(message):
    """Prints a message with a Rome timestamp."""
    print(f"[{get_timestamp_in_rome()}] {message}")

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
        device_map="auto"
    )
    return model, tokenizer

# --- Text Generation ---
def generate_solution(model, tokenizer, problem_text, logits_processors=None):
    """
    Generates a step-by-step solution for a given math problem.
    Uses a one-shot example and a system prompt for robust format control.
    """
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
        logits_processor=logits_processors
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    Compares two answers for numerical equivalence, ignoring formatting.
    """
    if ground_truth_answer is None or generated_answer is None:
        return False

    # Normalize by extracting all numerical parts (handles decimals, commas, negatives)
    gt_nums = re.findall(r'-?\\d*\\.?\\d+', str(ground_truth_answer))
    gen_nums = re.findall(r'-?\\d*\\.?\\d+', str(generated_answer))

    if not gt_nums or not gen_nums:
        # If no numbers found, fall back to case-insensitive string comparison
        return str(ground_truth_answer).strip().lower() == str(generated_answer).strip().lower()

    # Compare the first extracted number
    try:
        return float(gt_nums[0]) == float(gen_nums[0])
    except (ValueError, IndexError):
        return False


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
    tl_model = HookedTransformer.from_pretrained(
        MODEL_ID,
        hf_model=hf_model, # Pass the pre-loaded model
        torch_dtype=torch.bfloat16,
        device=DEVICE,
        trust_remote_code=True
    )
    print("TransformerLens model loaded.")
    return tl_model

def get_patching_results(
    model, 
    source_text, 
    destination_text,
    source_answer,
    destination_answer,
    excel_path  # Add excel_path parameter
):
    """
    Performs activation patching between a clean and corrupted run.
    Measures the change in logit difference for the correct vs. incorrect next token.
    """
    # 1. Tokenize all inputs
    source_tokens = model.to_tokens(source_text)
    dest_tokens = model.to_tokens(destination_text)
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
                "layer": layer, "component": f"Head {head}",
                "logit_diff_change": (patched_logit_diff - original_logit_diff).item()
            })
            
    # MLP Layers
    for layer in tqdm(range(model.cfg.n_layers), desc="Patching MLP Layers"):
        hook_name = get_act_name('mlp_out', layer)
        hook_fn = lambda act, hook: patching_hook(act, hook)
        patched_logits = model.run_with_hooks(dest_tokens, fwd_hooks=[(hook_name, hook_fn)])
        patched_logit_diff = get_logit_diff(patched_logits, source_ans_token, dest_ans_token)
        results.append({
            "layer": layer, "component": "MLP",
            "logit_diff_change": (patched_logit_diff - original_logit_diff).item()
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
    """
    # Placeholder for the actual API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print_timestamped_message("Warning: ANTHROPIC_API_KEY not set. Returning a placeholder error.")
        # Return the last sentence as a fallback for testing without an API key
        return incorrect_cot.strip().split('.')[-2] + '.'

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
        return error_sentence
    except Exception as e:
        print_timestamped_message(f"An error occurred with the Anthropic API: {e}")
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


def run_control_experiment(error_analysis_df, model, tokenizer, intervention_phrases, excel_path):
    """
    Runs the control experiment by injecting phrases at the point of error.
    """
    analyzable_df = error_analysis_df.dropna(subset=['error_sentence', 'full_generated_solution']).copy()
    analyzable_df = analyzable_df[~analyzable_df['error_sentence'].str.contains("ANTHROPIC_API_KEY not set")]
    
    if analyzable_df.empty:
        print_timestamped_message("No analyzable errors found. Skipping control experiment.")
        save_to_excel(pd.DataFrame(), "Control_Experiment", excel_path)
        return pd.DataFrame()

    intervention_results = []

    for _, row in tqdm(analyzable_df.iterrows(), total=len(analyzable_df), desc="Running Control Experiment"):
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
                is_corrected = is_correct(row['ground_truth_answer'], intervened_answer)
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
                'error': error_message
            })

    intervention_df = pd.DataFrame(intervention_results)
    save_to_excel(intervention_df, "Control_Experiment", excel_path)

    # Print summary
    correction_rates = intervention_df.groupby('condition')['is_corrected'].mean() * 100
    print_timestamped_message("Control experiment complete. Correction rates:")
    print(correction_rates.to_string(float_format="%.2f%%"))

    return intervention_df
