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
    core.print_timestamped_message("--- PRODUCTION MODE: Using the full dataset ---")
    dataset = full_dataset

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
# --- 6. Control Experiment (Reverse Lanham Test) ---
# We test if we can correct the model's reasoning by injecting a 'Corrective' phrase 
# right before the identified error. We compare this to 'Neutral' and 'Confirmation' phrases.
# Results are saved to the 'Control_Experiment' sheet.

# Define our experimental conditions
intervention_phrases = {
    "Corrective": " Wait, let me re-evaluate that.",
    "Neutral": " And,",
    "Confirmation": " Continuing,"
}

intervention_df = core.run_control_experiment(error_analysis_df, model, tokenizer, intervention_phrases, EXCEL_RESULTS_PATH)
if not intervention_df.empty:
    display(intervention_df.head())


# %%
# --- 7. Activation Patching Setup ---
# We now prepare for the main causal analysis. We need to find a 'clean' run (where our 
# 'Corrective' intervention worked) and a 'corrupted' run (where the 'Confirmation' 
# intervention failed on the same problem).

core.print_timestamped_message("Setting up activation patching experiment...")
source_prompt, destination_prompt, source_answer_text, destination_answer_text = [None] * 4

if not intervention_df.empty:
    pivot_df = intervention_df.pivot(index='problem', columns='condition', values='is_corrected')
    
    # Ensure required columns exist before filtering
    required_cols = ['Corrective', 'Confirmation']
    if all(col in pivot_df.columns for col in required_cols):
        successful_candidates = pivot_df[(pivot_df['Corrective'] == True) & (pivot_df['Confirmation'] == False)]
    else:
        successful_candidates = pd.DataFrame()

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

        source_prompt = truncated_cot + intervention_phrases["Corrective"]
        destination_prompt = truncated_cot + intervention_phrases["Confirmation"]

        source_full_cot = intervention_df[(intervention_df['problem'] == target_problem) & (intervention_df['condition'] == 'Corrective')].iloc[0]['intervened_cot']
        destination_full_cot = intervention_df[(intervention_df['problem'] == target_problem) & (intervention_df['condition'] == 'Confirmation')].iloc[0]['intervened_cot']

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
        excel_path=EXCEL_RESULTS_PATH
    )
    
    # --- 9. Visualizing Patching Results ---
    core.print_timestamped_message("Visualizing patching results...")
    
    # Separate attention and MLP data for visualization
    attention_results = patching_results_df[patching_results_df['component'] == 'Attention']
    mlp_results = patching_results_df[patching_results_df['component'] == 'MLP']

    # Create heatmap for Attention Heads
    if not attention_results.empty:
        fig_attn = px.imshow(
            attention_results.pivot(index='layer', columns='head', values='logit_diff_change'),
            labels=dict(x="Head", y="Layer", color="Logit Diff Change"),
            title="Activation Patching: Attention Heads",
            color_continuous_scale="RdBu",
            zmid=0
        )
        fig_attn.show()

    # Create bar chart for MLP Layers
    if not mlp_results.empty:
        fig_mlp = px.bar(
            mlp_results,
            x='layer',
            y='logit_diff_change',
            labels=dict(x="Layer", y="Logit Diff Change"),
            title="Activation Patching: MLP Layers"
        )
        fig_mlp.show()

core.print_timestamped_message("Script finished.")

