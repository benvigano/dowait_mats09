
# %%
# --- 1. Setup and Imports ---
from dotenv import load_dotenv
from importlib import reload
import torch
from datasets import load_dataset
import pandas as pd
from IPython.display import display
import plotly.express as px
import os

# ============================================================
# EXPERIMENT CONFIGURATION & CONSTANTS
# ============================================================

# Model Configuration
MODELS_TO_TEST = [
    {"type": "huggingface", "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
    {"type": "nebius", "id": "Qwen/Qwen3-14B"}
]

# Experiment Parameters
DEBUG_MODE = False
SAMPLE_SIZE = 10  # Number of examples to use in debug mode
PROD_SAMPLE_SIZE = 500  # Limit the number of examples for the full run
GENERALIZATION_SAMPLE_SIZE = 2  # Max incorrect problems to test for generalization. Set to None to use all.
MAX_STEERING_VALIDATION_SAMPLES = 1  # Max validation samples for steering experiment (None for no limit)

# Removed: Steering & Activation Patching functionality

# Current experiment: Only testing the original "Wait" intervention
CURRENT_INTERVENTIONS = {
    "Corrective_Original": "\nWait! ",
    "Corrective_Strong": " Wait, I made a mistake. ",
}

# Import our custom modules
import core
reload(core)

# Load environment variables from .env file
load_dotenv()
core.print_timestamped_message("Environment variables loaded. Script starting.")

# Define the path for our results folder
RESULTS_BASE_DIR = "results"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, f"experiment_{core.get_timestamp_in_rome().replace(' ', '_').replace(':', '-')}")
os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)
core.print_timestamped_message(f"Results will be saved to: {EXPERIMENT_RESULTS_DIR}")


# %%
# ============================================================
# PHASE 1: PROBLEM SELECTION (NEW: DIVERSE PROBLEMS FIRST!)
# ============================================================

core.print_timestamped_message("")
core.print_timestamped_message("=" * 60)
core.print_timestamped_message("🎲 SELECTING DIVERSE PROBLEMS")
core.print_timestamped_message("=" * 60)

# NEW: Select diverse problems from all subjects and difficulty levels
# Higher levels get more weight since they're more challenging and interesting
selected_problems = core.select_diverse_problems(PROD_SAMPLE_SIZE, DEBUG_MODE)

core.print_timestamped_message(f"📊 Selected {len(selected_problems)} problems for experiment")

# %%
# ============================================================  
# PHASE 2: MULTI-MODEL EXPERIMENT LOOP
# ============================================================

# Convert selected problems to dataset-like format for compatibility
class ProblemDataset:
    def __init__(self, problems):
        self.problems = problems
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        return self.problems[idx]

dataset = ProblemDataset(selected_problems)

# Store results for comparison
all_model_results = {}
all_patching_results = {}

for model_config in MODELS_TO_TEST:
    model_type = model_config["type"]
    model_id = model_config["id"]
    
    core.print_timestamped_message("")
    core.print_timestamped_message("=" * 80)
    core.print_timestamped_message(f"🚀 STARTING EXPERIMENTS FOR MODEL: {model_id} ({model_type})")
    core.print_timestamped_message("=" * 80)
    
    # Create model-specific results directory
    model_name = model_id.replace("/", "_").replace("-", "_")
    model_results_dir = os.path.join(EXPERIMENT_RESULTS_DIR, f"model_{model_name}")
    os.makedirs(model_results_dir, exist_ok=True)
    
    # ============================================================
    # PHASE 2A: SETUP (MODEL, TOKENIZER)
    # ============================================================
    core.print_timestamped_message(f"Loading model: {model_id} via {model_type}")
    from models import create_model
    model = create_model(model_type, model_id)
    core.print_timestamped_message(f"✅ {model_type.title()} model loaded")

    # ============================================================
    # PHASE 2B: BASELINE EXPERIMENT  
    # ============================================================
    core.print_timestamped_message("")
    core.print_timestamped_message("=" * 60)
    core.print_timestamped_message("🎯 STARTING BASELINE EXPERIMENT")
    core.print_timestamped_message("=" * 60)

    results_df = core.run_baseline_experiment(dataset, model, None, model_results_dir, model.get_model_id())
    core.print_timestamped_message(f"✅ Baseline experiment completed: {len(results_df)} problems processed")

    # Print database summary
    from database import print_database_summary
    print_database_summary(model_results_dir)

    # ============================================================
    # PHASE 2C: INTERVENTION TESTING
    # ============================================================
    core.print_timestamped_message("")
    core.print_timestamped_message("=" * 60)
    core.print_timestamped_message("💉 STARTING INTERVENTION TESTING")
    core.print_timestamped_message("=" * 60)

    # Use intervention phrases from local constants
    intervention_phrases = CURRENT_INTERVENTIONS

    intervention_df = core.run_insertion_test(results_df, model, intervention_phrases, model_results_dir)
    if not intervention_df.empty:
        core.print_timestamped_message(f"✅ Intervention testing completed: {len(intervention_df)} interventions tested")

    # Print updated database summary
    print_database_summary(model_results_dir)

    # ============================================================
    # PHASE 2D: ACTIVATION PATCHING (CAUSAL TRACING)
    # ============================================================
    core.print_timestamped_message("")
    core.print_timestamped_message("=" * 60)
    core.print_timestamped_message("🔍 STARTING ACTIVATION PATCHING")
    core.print_timestamped_message("=" * 60)

    # --- Define Patching Setup Using Real Problems ---
    # We'll use intervention data to create realistic clean/corrupted pairs
    patching_setup = {
        "use_intervention_data": True,
        "intervention_df": intervention_df,
        "patching_components": ["resid_pre", "attn_out", "mlp_out", "z"],
        "max_problems_to_patch": 5  # Limit for computational efficiency
    }

    # --- Run Activation Patching Experiment ---
    # Only run activation patching for HuggingFace models (TransformerLens requirement)
    if model_type == "huggingface":
        from high_level import run_activation_patching_experiment
        patching_results, patching_fig = run_activation_patching_experiment(
            model, patching_setup, model_results_dir
        )

        if patching_results:
            core.print_timestamped_message("Activation patching completed and saved to results directory.")
            
            # Store results for comparison
            all_patching_results[model_id] = patching_results
            
            # Print summary statistics for each component
            for component, data in patching_results.items():
                import numpy as np
                data_array = np.array(data)
                max_recovery = np.max(data_array)
                mean_recovery = np.mean(data_array)
                core.print_timestamped_message(f"  {component}: Max recovery = {max_recovery:.3f}, Mean recovery = {mean_recovery:.3f}")
                
                # Find the location with maximum recovery
                max_pos = np.unravel_index(np.argmax(data_array), data_array.shape)
                if component == "z" and len(data_array.shape) == 3:
                    core.print_timestamped_message(f"    Max recovery at Layer {max_pos[0]}, Head {max_pos[1]}, Position {max_pos[2]}")
                else:
                    core.print_timestamped_message(f"    Max recovery at Layer {max_pos[0]}, Position {max_pos[1]}")
    else:
        core.print_timestamped_message("⚠️ Activation patching skipped for API-based models (TransformerLens not supported)")
        patching_results = None

    # ============================================================
    # PHASE 2E: STEERING VECTOR EXPERIMENT
    # ============================================================
    steering_df = None
    if model_type == "huggingface" and not intervention_df.empty:
        core.print_timestamped_message("")
        core.print_timestamped_message("=" * 60)
        core.print_timestamped_message("🎯 STARTING STEERING VECTOR EXPERIMENT")
        core.print_timestamped_message("=" * 60)
        
        from high_level import run_steering_experiment
        steering_df = run_steering_experiment(
            model, results_df, intervention_df, intervention_phrases, model_results_dir, 
            layer=21, max_validation_samples=MAX_STEERING_VALIDATION_SAMPLES
        )
        
        if steering_df is not None and not steering_df.empty:
            steering_success = (steering_df['is_corrected_by_steering'].sum() / len(steering_df)) * 100
            core.print_timestamped_message(f"✅ Steering experiment completed: {steering_success:.1f}% success rate")
        else:
            core.print_timestamped_message("⚠️ Steering experiment could not be completed")
    else:
        core.print_timestamped_message("⚠️ Steering experiment skipped (API model or no interventions)")

    # Store model results
    all_model_results[model_id] = {
        'model_type': model_type,
        'baseline_accuracy': (results_df['is_correct'].sum() / len(results_df)) * 100 if not results_df.empty else 0,
        'intervention_success': (intervention_df['is_corrected'].sum() / len(intervention_df)) * 100 if not intervention_df.empty else 0,
        'patching_results': patching_results,
        'steering_success': (steering_df['is_corrected_by_steering'].sum() / len(steering_df)) * 100 if steering_df is not None and not steering_df.empty else None
    }
    
    # Clean up model from memory before loading next one
    from models import drop_model_from_memory
    model = drop_model_from_memory(model)

# ============================================================
# PHASE 3: CROSS-MODEL COMPARISON
# ============================================================
core.print_timestamped_message("")
core.print_timestamped_message("=" * 80)
core.print_timestamped_message("📊 CROSS-MODEL COMPARISON SUMMARY")
core.print_timestamped_message("=" * 80)

for model_id, results in all_model_results.items():
    core.print_timestamped_message(f"\n{model_id} ({results['model_type']}):")
    core.print_timestamped_message(f"  Baseline Accuracy: {results['baseline_accuracy']:.1f}%")
    core.print_timestamped_message(f"  Intervention Success: {results['intervention_success']:.1f}%")
    
    if results['patching_results']:
        for component, data in results['patching_results'].items():
            import numpy as np
            max_recovery = np.max(np.array(data))
            core.print_timestamped_message(f"  {component} Max Recovery: {max_recovery:.3f}")
    else:
        core.print_timestamped_message(f"  Activation Patching: Not available (API model)")
    
    if results['steering_success'] is not None:
        core.print_timestamped_message(f"  Steering Vector Success: {results['steering_success']:.1f}%")
    else:
        core.print_timestamped_message(f"  Steering Vector: Not available")

# Save comparison data
comparison_file = os.path.join(EXPERIMENT_RESULTS_DIR, "model_comparison.json")
import json
with open(comparison_file, 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for model_id, results in all_model_results.items():
        serializable_results[model_id] = {
            'model_type': results['model_type'],
            'baseline_accuracy': results['baseline_accuracy'],
            'intervention_success': results['intervention_success'],
            'patching_results': {k: v.tolist() if hasattr(v, 'tolist') else v 
                               for k, v in results['patching_results'].items()} if results['patching_results'] else None,
            'steering_success': results['steering_success']
        }
    json.dump(serializable_results, f, indent=4)

core.print_timestamped_message(f"Cross-model comparison saved to: {comparison_file}")


# %%
# ============================================================
# EXPERIMENT COMPLETE
# ============================================================
core.print_timestamped_message("")
core.print_timestamped_message("=" * 60)
core.print_timestamped_message("✅ EXPERIMENT COMPLETE")
core.print_timestamped_message("=" * 60)

# Print final database summary
print_database_summary(EXPERIMENT_RESULTS_DIR)

core.print_timestamped_message("Script finished.")