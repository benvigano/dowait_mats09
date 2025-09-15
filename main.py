
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
    #{"type": "huggingface", "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
    {"type": "huggingface", "id": "Qwen/Qwen3-1.7B"},  # Local inference
    #{"type": "huggingface", "id": "Qwen/Qwen3-14B"}  # Official Qwen3-14B for baseline/intervention experiments
]

# Experiment Parameters
DEBUG_MODE = False
SAMPLE_SIZE = 10  # Number of examples to use in debug mode
PROD_SAMPLE_SIZE = 300  # Limit the number of examples for the full run

# Activation patching sample caps (for testing)
ACTIVATION_PATCHING_MAX_TRAIN_SAMPLES = None
ACTIVATION_PATCHING_MAX_VALIDATION_SAMPLES = None

# Try different layers and strengths
steering_configs = [
    {"layer": 16, "strength": 2.0},
    {"layer": 20, "strength": 1.0},
    {"layer": 24, "strength": 0.5},
]

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

    baseline_df = core.run_baseline_experiment(dataset, model, None, model_results_dir, model.get_model_id())
    core.print_timestamped_message(f"✅ Baseline experiment completed: {len(baseline_df)} problems processed")

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

    intervention_df = core.run_insertion_test(baseline_df, model, intervention_phrases, model_results_dir)
    if not intervention_df.empty:
        core.print_timestamped_message(f"✅ Intervention testing completed: {len(intervention_df)} interventions tested")

    # Print updated database summary
    print_database_summary(model_results_dir)

    # ============================================================
    # PHASE 2D: NNSIGHT STEERING VECTOR EXPERIMENT  
    # ============================================================
    steering_df = None
    # Run steering for HuggingFace models that support nnsight
    if model_type == "huggingface" and hasattr(model, 'get_nnsight_model'):
        core.print_timestamped_message("")
        core.print_timestamped_message("=" * 60)
        core.print_timestamped_message("🎯 STARTING NNSIGHT STEERING EXPERIMENT")
        core.print_timestamped_message(f"Using {model_id} for steering")
        core.print_timestamped_message("=" * 60)
        
        # --- AGGRESSIVE MEMORY CLEANUP ---
        # CRITICAL: Drop the main model and clear all caches before spawning worker processes.
        core.print_timestamped_message("🗑️ Aggressively clearing memory before starting steering workers...")
        from models import drop_model_from_memory
        import gc
        
        # Get model ID before dropping
        model_id_for_steering = model.get_model_id()
        
        # Drop the model and clear references
        model = drop_model_from_memory(model)
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        core.print_timestamped_message("✅ Memory cleared. Starting steering experiment...")
        
        # --- RUN STEERING EXPERIMENT IN ISOLATED PROCESSES ---
        from nnsight_steering import run_nnsight_steering_experiment
        from models import HuggingFaceModel

        # Create steering model interface using the same model as baseline/intervention
        steering_model_interface = HuggingFaceModel(model_id, load_now=False)
        
        best_steering_df = None
        best_success_rate = 0
        
        for config in steering_configs:
            core.print_timestamped_message(f"Trying layer {config['layer']} with strength {config['strength']}")
            
            steering_df = run_nnsight_steering_experiment(
                model_interface=steering_model_interface,
                results_dir=model_results_dir,
                layer=config["layer"],
                steering_strength=config["strength"],
                baseline_data=baseline_df,  # Pass baseline data directly
                intervention_data=intervention_df,  # Pass intervention data directly
                max_train_samples=ACTIVATION_PATCHING_MAX_TRAIN_SAMPLES,
                max_validation_samples=ACTIVATION_PATCHING_MAX_VALIDATION_SAMPLES
            )
            
            if steering_df is not None and not steering_df.empty:
                success_rate = (steering_df['is_corrected'].sum() / len(steering_df)) * 100
                core.print_timestamped_message(f"Layer {config['layer']}, strength {config['strength']}: {success_rate:.1f}% success rate")
                
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_steering_df = steering_df
                    best_config = config
            else:
                core.print_timestamped_message(f"Layer {config['layer']}, strength {config['strength']}: Failed")
        
        steering_df = best_steering_df
        
        if steering_df is not None and not steering_df.empty:
            best_config = best_config if 'best_config' in locals() else {"layer": 20, "strength": 1.0}
            steering_success = (steering_df['is_corrected'].sum() / len(steering_df)) * 100
            core.print_timestamped_message(f"✅ NNsight steering completed: {steering_success:.1f}% success rate")
            
            # Save results to database
            from database import insert_nnsight_steering_result
            import torch
            insert_nnsight_steering_result(
                results_dir=model_results_dir,
                model_id=model_id_for_steering,
                layer=best_config["layer"],
                steering_strength=best_config["strength"],
                train_pairs_count=len(steering_df),  # This will be corrected in the actual implementation
                validation_pairs_count=len(steering_df),
                success_rate=steering_success,
                successful_corrections=int(steering_df['is_corrected'].sum()),
                steering_vector_norm=1.0,  # This will be corrected in the actual implementation  
                validation_results=steering_df.to_dict('records')
            )
            
            # Save detailed results
            steering_file = os.path.join(model_results_dir, "nnsight_steering_results.json")
            steering_df.to_json(steering_file, orient='records', indent=4)
            core.print_timestamped_message(f"Steering results saved to: {steering_file}")
            
        else:
            core.print_timestamped_message("⚠️ NNsight steering experiment could not be completed")
    else:
        core.print_timestamped_message("⚠️ Steering experiment skipped: model is not Qwen/Qwen3-14B.")


    # Store model results
    steering_success_rate = None
    if steering_df is not None and not steering_df.empty:
        steering_success_rate = (steering_df['is_corrected'].sum() / len(steering_df)) * 100
    
    all_model_results[model_id] = {
        'model_type': model_type,
        'baseline_accuracy': (baseline_df['is_correct'].sum() / len(baseline_df)) * 100 if not baseline_df.empty else 0,
        'intervention_success': (intervention_df['is_corrected'].sum() / len(intervention_df)) * 100 if not intervention_df.empty else 0,
        'nnsight_steering_success': steering_success_rate
    }
    
    # Clean up model from memory before loading next one
    from models import drop_model_from_memory
    model = drop_model_from_memory(model)



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