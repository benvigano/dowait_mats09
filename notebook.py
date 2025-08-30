# %%
# --- 1. Setup and Imports ---
from dotenv import load_dotenv
from importlib import reload
import torch
from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display
import plotly.express as px
import os

# Import our custom module
import core
reload(core)

# Load environment variables from .env file
load_dotenv()
core.print_timestamped_message("Environment variables loaded. Script starting.")

# Define the path for our results file
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
EXCEL_RESULTS_PATH = os.path.join(RESULTS_DIR, f"experiment_results_{core.get_timestamp_in_rome().replace(' ', '_').replace(':', '-')}.xlsx")
core.print_timestamped_message(f"Results will be saved to: {EXCEL_RESULTS_PATH}")


# %%
# ============================================================
# PHASE 1: SETUP (MODEL, TOKENIZER, DATASET)
# ============================================================

# --- 2. Load Model and Tokenizer ---
# Load HuggingFace model for generation
model, tokenizer = core.load_model_and_tokenizer()
core.print_timestamped_message("✅ HuggingFace model and tokenizer loaded")

# --- 3. Load Dataset ---
core.print_timestamped_message("Loading dataset...")
full_dataset = load_dataset("EleutherAI/hendrycks_math", "algebra", split='test')

# In debug mode, we'll use a small, deterministic sample. Otherwise, use the full dataset.
if core.DEBUG_MODE:
    core.print_timestamped_message(f"--- DEBUG MODE: Using a sample of {core.SAMPLE_SIZE} examples ---")
    dataset = full_dataset.select(range(core.SAMPLE_SIZE))
else:
    core.print_timestamped_message(f"--- PRODUCTION MODE: Limiting to {core.PROD_SAMPLE_SIZE} examples ---")
    dataset = full_dataset.select(range(core.PROD_SAMPLE_SIZE))

display(dataset)
core.print_timestamped_message("Dataset loaded.")

# %%
# ============================================================
# PHASE 1: BASELINE EXPERIMENT
# ============================================================
core.print_timestamped_message("=" * 60)
core.print_timestamped_message("🎯 STARTING BASELINE EXPERIMENT")
core.print_timestamped_message("=" * 60)

results_df = core.run_baseline_experiment(dataset, model, tokenizer, EXCEL_RESULTS_PATH)
core.print_timestamped_message(f"✅ Baseline experiment completed: {len(results_df)} problems processed")


# %%
# ============================================================
# PHASE 2: ERROR IDENTIFICATION  
# ============================================================
core.print_timestamped_message("")
core.print_timestamped_message("=" * 60)
core.print_timestamped_message("🔍 STARTING ERROR IDENTIFICATION")
core.print_timestamped_message("=" * 60)

error_analysis_df = core.identify_errors(results_df, EXCEL_RESULTS_PATH, tokenizer=tokenizer, use_token_indexing=True)
if not error_analysis_df.empty:
    core.print_timestamped_message(f"✅ Error identification completed: {len(error_analysis_df)} problems analyzed")


# %%
# ============================================================
# PHASE 3: INTERVENTION TESTING
# ============================================================
core.print_timestamped_message("")
core.print_timestamped_message("=" * 60)
core.print_timestamped_message("💉 STARTING INTERVENTION TESTING")
core.print_timestamped_message("=" * 60)

# Use intervention phrases from core constants
intervention_phrases = core.CURRENT_INTERVENTIONS

intervention_df = core.run_insertion_test(error_analysis_df, model, tokenizer, intervention_phrases, EXCEL_RESULTS_PATH)
if not intervention_df.empty:
    core.print_timestamped_message(f"✅ Intervention testing completed: {len(intervention_df)} interventions tested")


# %%
# --- 6.5. DEBUG: Ensure Patching Candidate ---
# In debug mode, we want to guarantee that the activation patching step runs.
# This block checks if a suitable clean/corrupted pair was found. If not,
# it injects a mock example to ensure the patching logic can be tested.
if core.DEBUG_MODE:
    core.print_timestamped_message("--- DEBUG: Checking for activation patching candidate ---")
    
    candidate_found = False
    if not intervention_df.empty:
        # We need to pivot to check conditions for the same problem
        try:
            pivot_df_check = intervention_df.pivot(index='problem', columns='condition', values='is_corrected')
            corrective_cols_check = [col for col in pivot_df_check.columns if col.startswith('Corrective')]
            
            if corrective_cols_check:
                for col in corrective_cols_check:
                    # A candidate is any problem where a corrective intervention worked
                    if not pivot_df_check[pivot_df_check[col] == True].empty:
                        candidate_found = True
                        core.print_timestamped_message("--- DEBUG: A valid patching candidate already exists. No mock data needed. ---")
                        break
        except Exception as e:
            core.print_timestamped_message(f"--- DEBUG: Could not pivot intervention_df to check for candidates, likely due to duplicate entries. Error: {e} ---")


    if not candidate_found:
        core.print_timestamped_message("--- DEBUG: No valid candidate found. Injecting mock data to ensure patching runs. ---")
        
        mock_problem_text = "This is a mock problem for testing activation patching. What is 2+2?"
        
        # Create a mock row for the error_analysis_df. It needs to contain all the columns
        # that the patching setup step will try to access.
        mock_error_row = pd.DataFrame([{
            'problem': mock_problem_text,
            'full_generated_solution': "I will calculate 2+2. The result is 5.",
            'error_sentence': "The result is 5.",
            'problem_id': 999,
            'ground_truth_full': 'The answer is \\boxed{4}',
            'is_correct': False,
            'ground_truth_answer': '4',
            'generated_answer': '5',
            'error': None,
            'error_identification_error': None
        }])
        
        # Create mock rows for the intervention_df to create the clean/corrupted pair.
        mock_intervention_rows = pd.DataFrame([
            {
                'problem': mock_problem_text,
                'condition': 'Corrective_Original',
                'is_corrected': True, # This is the "clean" run that worked
                'intervened_cot': "I will calculate 2+2. Wait, let me re-evaluate that. The correct answer is 4. \\boxed{4}.",
                'intervened_answer': '4',
                'problem_id': 999,
                'error': None
            }
        ])

        # Safely append the mock data. `pd.concat` is the modern way.
        if not error_analysis_df.empty:
            error_analysis_df = pd.concat([error_analysis_df, mock_error_row], ignore_index=True)
        else:
            error_analysis_df = mock_error_row

        if not intervention_df.empty:
            intervention_df = pd.concat([intervention_df, mock_intervention_rows], ignore_index=True)
        else:
             intervention_df = mock_intervention_rows
        
        core.print_timestamped_message("--- DEBUG: Mock data injected successfully. ---")


# %%
# --- 7. Activation Patching Setup ---
# We now prepare for the main causal analysis. We need to find a 'clean' run (where our 
# 'Corrective' intervention worked) and a 'corrupted' run (where the original uninformed 
# continuation failed on the same problem).

# %%
# ============================================================
# PHASE 4: ACTIVATION PATCHING ANALYSIS
# ============================================================
# Load TransformerLens model for activation patching
tl_model = core.load_tl_model(model)
core.print_timestamped_message("Setting up activation patching experiment...")

if not intervention_df.empty:
    pivot_df = intervention_df.pivot(index='problem', columns='condition', values='is_corrected')
    
    # Look for any corrective condition that successfully fixed the error
    corrective_cols = [col for col in pivot_df.columns if col.startswith('Corrective')]
    
    successful_candidates = pd.DataFrame()
    best_corrective_condition = None
    
    if corrective_cols:
        # Find any problem where a Corrective intervention succeeded
        for corrective_col in corrective_cols:
            candidates = pivot_df[pivot_df[corrective_col] == True]
            if not candidates.empty:
                successful_candidates = candidates
                best_corrective_condition = corrective_col
                core.print_timestamped_message(f"Found {len(candidates)} candidates with {corrective_col}")
                break

    if not successful_candidates.empty:
        target_problem = successful_candidates.index[0]
        core.print_timestamped_message(f"Found a perfect candidate for patching: {target_problem[:80]}...")
        
        # Reconstruct prompts from our data
        analyzable_row = error_analysis_df[error_analysis_df['problem'] == target_problem].iloc[0]
        original_cot = analyzable_row['full_generated_solution']
        mistake_sentence = analyzable_row['mistake_sentence']
        mistake_index = original_cot.find(mistake_sentence)
        truncated_cot = original_cot[:mistake_index]

        # Define source (clean) and destination (corrupted) prompts
        source_prompt = truncated_cot + intervention_phrases[best_corrective_condition]
        destination_prompt = truncated_cot  # Original uninformed continuation

        source_full_cot = intervention_df[(intervention_df['problem'] == target_problem) & (intervention_df['condition'] == best_corrective_condition)].iloc[0]['intervened_cot']
        destination_full_cot = original_cot

        core.print_timestamped_message(f"Using corrective condition: {best_corrective_condition}")
        core.print_timestamped_message("Comparing corrected intervention vs. original uninformed continuation")

        # Define correct and incorrect answers for logit diff calculation
        source_answer_text = core.extract_boxed_answer(source_full_cot)
        destination_answer_text = core.extract_boxed_answer(destination_full_cot)

        # --- 8. Run Activation Patching ---
        patching_results_df = core.get_patching_results(
            tl_model, 
            source_prompt, 
            destination_prompt,
            source_answer_text,
            destination_answer_text,
            excel_path=EXCEL_RESULTS_PATH,
            problem_id=analyzable_row['problem_id'],
            corrective_condition=best_corrective_condition
        )
        
        # --- 9. Visualize Results ---
        if not patching_results_df.empty:
            # Separate attention and MLP data for visualization
            attention_results = patching_results_df[patching_results_df['component'].str.contains('Head')]
            mlp_results = patching_results_df[patching_results_df['component'] == 'MLP']

            # Summarize Attention Heads results
            if not attention_results.empty:
                print(f"Attention heads results: {len(attention_results)} components analyzed")
                print(f"Top 5 attention heads by logit diff change:")
                top_attn = attention_results.nlargest(5, 'logit_diff_change')[['layer', 'component', 'logit_diff_change']]
                print(top_attn.to_string(index=False))

            # Summarize MLP Layers results
            if not mlp_results.empty:
                print(f"\nMLP layers results: {len(mlp_results)} layers analyzed")
                print(f"Top 5 MLP layers by logit diff change:")
                top_mlp = mlp_results.nlargest(5, 'logit_diff_change')[['layer', 'component', 'logit_diff_change']]
                print(top_mlp.to_string(index=False))

core.print_timestamped_message("Script finished.")

