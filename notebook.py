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
# --- 2. Model and Tokenizer Loading ---
model, tokenizer = core.load_model_and_tokenizer()
core.print_timestamped_message("Model and tokenizer loaded.")


# %%
# --- 3. Dataset Loading ---
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
# --- 4. Baseline Experiment ---
# We generate solutions for the dataset without any interventions to see the model's default performance.
# All results, including errors, are saved to the 'Baseline_Results' sheet in our Excel file.
results_df = core.run_baseline_experiment(dataset, model, tokenizer, EXCEL_RESULTS_PATH)
display(results_df.head())


# %%
# --- 5. Error Identification ---
# For the examples the model got wrong, we use an external LLM to find the specific sentence where the error occurs.
# Results are saved to the 'Error_Analysis' sheet.
error_analysis_df = core.identify_errors(results_df, EXCEL_RESULTS_PATH)
if not error_analysis_df.empty:
    display(error_analysis_df.head())


# %%
# --- 6. Insertion Test (Reverse Lanham Test) ---
# We test if we can correct the model's reasoning by injecting a 'Corrective' phrase 
# right before the identified error. We compare this to 'Neutral' and 'Confirmation' phrases.
# Results are saved to the 'Insertion_Test' sheet.

# Define our experimental conditions
intervention_phrases = {
    "Corrective_Strong": " Wait, I made a mistake.",
    "Corrective_Alert": " Hold on!",
    "Corrective_Reconsider": " Wait, I need to reconsider.",
    "Corrective_Wrong": " Wait, that was wrong!",
    "Corrective_Original": " Wait, let me re-evaluate that.",
    "Neutral": " And,",
    "Confirmation": " Continuing,"
}

intervention_df = core.run_insertion_test(error_analysis_df, model, tokenizer, intervention_phrases, EXCEL_RESULTS_PATH)
if not intervention_df.empty:
    display(intervention_df.head())


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
# 'Corrective' intervention worked) and a 'corrupted' run (where the 'Confirmation' 
# intervention failed on the same problem).

core.print_timestamped_message("Setting up activation patching experiment...")
source_prompt, destination_prompt, source_answer_text, destination_answer_text = [None] * 4

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
        
        # Reconstruct the prompts and answers from our previous experimental data
        # Find the original generation data for this specific problem
        analyzable_row = error_analysis_df[error_analysis_df['problem'] == target_problem].iloc[0]
        original_cot = analyzable_row['full_generated_solution']
        error_sentence = analyzable_row['error_sentence']
        error_index = original_cot.find(error_sentence)
        truncated_cot = original_cot[:error_index]

        source_prompt = truncated_cot + intervention_phrases[best_corrective_condition]
        destination_prompt = truncated_cot  # Original uninformed continuation (no intervention)

        source_full_cot = intervention_df[(intervention_df['problem'] == target_problem) & (intervention_df['condition'] == best_corrective_condition)].iloc[0]['intervened_cot']
        destination_full_cot = original_cot  # Use the original incorrect reasoning
        
        core.print_timestamped_message(f"Using corrective condition: {best_corrective_condition}")
        core.print_timestamped_message("Comparing corrected intervention vs. original uninformed continuation")

        source_answer_text = source_full_cot[len(source_prompt):].strip()
        destination_answer_text = destination_full_cot[len(destination_prompt):].strip()
        
        print(f"Source (Clean) Prompt: '{source_prompt}'")
        print(f"Destination (Corrupted) Prompt: '{destination_prompt}'")
    else:
        core.print_timestamped_message("Could not find a perfect clean/corrupted pair. Skipping activation patching.")
else:
    core.print_timestamped_message("No intervention data available. Skipping activation patching.")


# %%
# --- 8. Running the Patching Experiment ---
# If a suitable pair was found, we load the model into TransformerLens and run the patching experiment.
# This identifies which model components (attention heads, MLPs) are causally responsible for the self-correction.
# Results are saved to the 'Activation_Patching' sheet.

if source_prompt:
    core.print_timestamped_message("Loading model into TransformerLens...")
    tl_model = core.load_tl_model(model)
    
    patching_results_df = core.get_patching_results(
        model=tl_model,
        source_text=source_prompt,
        destination_text=destination_prompt,
        source_answer=source_answer_text,
        destination_answer=destination_answer_text,
        excel_path=EXCEL_RESULTS_PATH,
        problem_id=target_problem,
        corrective_condition=best_corrective_condition
    )
    
    # --- 9. Visualizing Patching Results ---
    core.print_timestamped_message("Visualizing patching results...")
    
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

