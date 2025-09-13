"""
High-level experiment workflows and main runners.
"""

import pandas as pd
import os
import json
import traceback
from tqdm import tqdm
import torch
import plotly.express as px
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

from database import insert_baseline_result, insert_intervention_result, insert_activation_patching_result, insert_steering_result

from cache import (
    print_timestamped_message, get_from_generation_cache, 
    get_from_generalization_cache, save_to_generalization_cache
)
from low_level import (
    solve_problem_baseline, solve_problem_with_intervention,
    extract_boxed_answer, is_correct, evaluate_answer,
    perform_activation_patching, calculate_steering_vector, apply_steering_vector,
    get_logit_diff
)

from prompts import create_baseline_prompt

# Note: Configuration constants moved to notebook.py

# --- Utilities ---

# Removed save_to_json and save_experiment_metadata - all data now stored in SQLite

# --- Main Experiment Functions ---

def run_baseline_experiment(dataset, model, tokenizer, results_dir, model_id=None):
    """
    Runs the baseline experiment with smart evaluation for mismatched answers.
    Uses Anthropic to properly categorize responses as correct/incorrect/failure.
    Saves results to SQLite database.
    """
    # --- 1. Separate cached and uncached problems ---
    uncached_examples = []
    cached_results = []
    
    print_timestamped_message("Checking cache for baseline results...")
    
    # Use a simple loop for the fast cache check instead of a tqdm bar
    for example in dataset:
        problem = example['problem']
        prompt = create_baseline_prompt(problem)
        cached_solution = get_from_generation_cache(prompt, model.get_model_id())
        
        if cached_solution is not None:
            ground_truth = example['solution']
            ground_truth_answer = extract_boxed_answer(ground_truth)
            
            # Use unified evaluation
            eval_result = evaluate_answer(problem, ground_truth_answer, cached_solution)
            
            # Reconstruct prompt for cached results
            raw_prompt = create_baseline_prompt(problem)
            
            cached_results.append({
                "problem": problem,
                "raw_prompt": raw_prompt,
                "ground_truth_full": ground_truth,
                "ground_truth_answer": ground_truth_answer,
                "full_generated_solution": cached_solution,
                "generated_answer": eval_result['generated_answer'],
                "is_correct": eval_result['is_correct'],
                "smart_evaluation": eval_result['evaluation_result'],
                "evaluation_method": eval_result['evaluation_method'],
                "error_line_number": eval_result.get('error_line_number'),
                "error_line_content": eval_result.get('error_line_content'),
                "mistake_sentence_usable": eval_result.get('mistake_sentence_usable', False),
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
        pbar = tqdm(uncached_examples, desc="Baseline Generation")
        for example in pbar:
            problem = example['problem']
            ground_truth = example['solution']
            # Define ground_truth_answer here, outside the try block
            ground_truth_answer = extract_boxed_answer(ground_truth)
            
            try:
                full_solution, raw_prompt = solve_problem_baseline(model, problem)
                
                # Use unified evaluation
                eval_result = evaluate_answer(problem, ground_truth_answer, full_solution)
                
                error_message = None
            except Exception as e:
                print_timestamped_message(f"ERROR generating solution for problem: {problem[:80]}... Details: {e}")
                full_solution = f"Error generating solution: {str(e)}"
                raw_prompt = create_baseline_prompt(problem)  # Still provide prompt for error cases
                eval_result = {
                    'is_correct': False,
                    'generated_answer': None,
                    'evaluation_method': 'generation_error',
                    'evaluation_result': 'failure'
                }
                error_message = traceback.format_exc()
            
            processed_results.append({
                "problem": problem,
                "raw_prompt": raw_prompt,
                "ground_truth_full": ground_truth,
                "ground_truth_answer": ground_truth_answer,
                "full_generated_solution": full_solution,
                "generated_answer": eval_result['generated_answer'],
                "is_correct": eval_result['is_correct'],
                "smart_evaluation": eval_result['evaluation_result'],
                "evaluation_method": eval_result['evaluation_method'],
                "error_line_number": eval_result.get('error_line_number'),
                "error_line_content": eval_result.get('error_line_content'),
                "mistake_sentence_usable": eval_result.get('mistake_sentence_usable', False),
                "error": error_message
            })
        pbar.close()

    # --- 3. Combine, save to SQLite, and summarize ---
    final_results_list = cached_results + processed_results
    
    # Add problem_id for traceability
    for i, result in enumerate(final_results_list):
        result['problem_id'] = i
        
    results_df = pd.DataFrame(final_results_list)
    
    # Save to SQLite database
    experiment_params = {
        'model_id': model_id,
        'temperature': 0.2,
        'max_tokens': 4096 * 2,
        'do_sample': True
    }
    
    print_timestamped_message("Saving baseline results to SQLite database...")
    for _, row in results_df.iterrows():
        insert_baseline_result(
            results_dir=results_dir,
            experiment_params=experiment_params,
            problem=row['problem'],
            ground_truth_answer=row['ground_truth_answer'],
            baseline_raw_response=row['full_generated_solution'],
            baseline_correct=1 if row['is_correct'] else 0,
            baseline_error_line_number=row.get('error_line_number'),
            baseline_error_line_content=row.get('error_line_content')
        )
    
    # Print summary with smart evaluation breakdown
    if not results_df.empty:
        accuracy = (results_df['is_correct'].sum() / len(results_df)) * 100
        errors = results_df['error'].count()
        
        # Calculate average reasoning steps
        step_counts = []
        for solution in results_df['full_generated_solution'].dropna():
            lines = [line.strip() for line in str(solution).split('\n') if line.strip()]
            step_counts.append(len(lines))
        
        avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
        
        # Smart evaluation breakdown
        eval_counts = results_df['smart_evaluation'].value_counts()
        method_counts = results_df['evaluation_method'].value_counts()
        
        print_timestamped_message(f"Baseline complete. Smart Accuracy: {accuracy:.2f}%. Generation errors: {errors}. Avg steps: {avg_steps:.1f}")
        print_timestamped_message(f"Smart evaluation breakdown: {dict(eval_counts)}")
        print_timestamped_message(f"Evaluation methods: {dict(method_counts)}")
        
    else:
        print_timestamped_message("Baseline complete. No results to analyze.")
    
    return results_df

# Removed extract_error_analysis_for_intervention - now integrated into run_insertion_test

def run_insertion_test(baseline_df, model, intervention_phrases, results_dir):
    """
    Runs the insertion test by injecting phrases at the point of mathematical errors.
    Directly uses baseline results to find problems with usable error locations.
    """
    print_timestamped_message("Starting intervention testing...")
    
    # --- 1. Filter for problems with usable error locations ---
    analyzable_df = baseline_df[
        (baseline_df['smart_evaluation'] == 'incorrect') & 
        (baseline_df['mistake_sentence_usable'] == True)
    ].copy()
    
    if analyzable_df.empty:
        print_timestamped_message("No problems with usable error locations found. Skipping intervention testing.")
        return pd.DataFrame()
    
    print_timestamped_message(f"Found {len(analyzable_df)} problems with usable error locations for intervention testing.")
    
    # Add mistake_sentence field for backward compatibility
    analyzable_df = analyzable_df.copy()
    analyzable_df['mistake_sentence'] = analyzable_df['error_line_content']

    # --- 2. Separate cached and uncached interventions ---
    uncached_interventions = []
    cached_results = []

    print_timestamped_message("Checking cache for intervention results...")
    
    for _, row in analyzable_df.iterrows():
        original_cot = row['full_generated_solution']
        mistake_sentence = row['mistake_sentence']
        
        # Determine truncation point - truncate AFTER the mistake sentence
        mistake_index = original_cot.find(mistake_sentence)
        if mistake_index == -1:
            continue # Should not happen due to pre-filtering
        
        # Find the end of the mistake sentence and truncate after it
        mistake_end_index = mistake_index + len(mistake_sentence)
        truncated_cot = original_cot[:mistake_end_index]

        for condition, phrase in intervention_phrases.items():
            # Create proper intervention prompt with full problem context
            try:
                from prompts import create_intervention_prompt
                prompt_with_intervention = create_intervention_prompt(
                    row['problem'], original_cot, mistake_sentence, phrase
                )
            except ValueError as e:
                print_timestamped_message(f"Could not create intervention prompt: {e}")
                continue
                
            cached_solution = get_from_generation_cache(prompt_with_intervention, model.get_model_id())
            
            if cached_solution is not None:
                ground_truth_answer = row['ground_truth_answer']
                eval_result = evaluate_answer(row['problem'], ground_truth_answer, cached_solution)
                
                cached_results.append({
                    'problem_id': row['problem_id'], 'problem': row['problem'], 'ground_truth_answer': ground_truth_answer,
                    'condition': condition, 'intervened_cot': cached_solution, 'intervened_answer': eval_result['generated_answer'],
                    'is_corrected': eval_result['is_correct'], 'error': None, 'mistake_sentence': mistake_sentence,
                    'truncation_point': mistake_end_index, 'intervention_phrase': phrase,
                    'original_cot_length': len(original_cot), 'truncated_cot_length': len(truncated_cot),
                    'final_prompt': prompt_with_intervention
                })
            else:
                uncached_interventions.append((row, condition, phrase, prompt_with_intervention, truncated_cot, mistake_end_index, original_cot))

    print_timestamped_message(f"Intervention Test: Found {len(cached_results)} cached results. Processing {len(uncached_interventions)} new generations.")

    # --- 3. Process uncached interventions ---
    processed_results = []
    if uncached_interventions:
        pbar_gen = tqdm(uncached_interventions, desc="Intervention Generation")
        for row, condition, phrase, prompt, truncated_cot, mistake_end_index, original_cot in pbar_gen:
            try:
                # Use the new intervention-specific function
                intervened_cot, full_prompt = solve_problem_with_intervention(
                    model, row['problem'], row['full_generated_solution'], 
                    row['mistake_sentence'], phrase
                )
                eval_result = evaluate_answer(row['problem'], row['ground_truth_answer'], intervened_cot)
                error_message = None
            except Exception as e:
                print_timestamped_message(f"ERROR during intervention for problem: {row['problem'][:80]}... Condition: {condition}. Details: {e}")
                intervened_cot = f"Error during generation: {str(e)}"
                eval_result = {
                    'is_correct': False,
                    'generated_answer': None,
                    'evaluation_method': 'generation_error',
                    'evaluation_result': 'failure'
                }
                error_message = traceback.format_exc()
                full_prompt = "Error occurred before prompt creation"

            processed_results.append({
                'problem_id': row['problem_id'], 'problem': row['problem'], 'ground_truth_answer': row['ground_truth_answer'],
                'condition': condition, 'intervened_cot': intervened_cot, 'intervened_answer': eval_result['generated_answer'],
                'is_corrected': eval_result['is_correct'], 'error': error_message, 'mistake_sentence': row['mistake_sentence'],
                'truncation_point': mistake_end_index, 'intervention_phrase': phrase,
                'original_cot_length': len(original_cot), 'truncated_cot_length': len(truncated_cot),
                'final_prompt': full_prompt
            })
        pbar_gen.close()

    # --- 4. Combine, save to SQLite, and summarize ---
    final_results_list = cached_results + processed_results
    intervention_df = pd.DataFrame(final_results_list)
    
    # Save intervention results to SQLite database
    print_timestamped_message("Saving intervention results to SQLite database...")
    for _, row in intervention_df.iterrows():
        insert_intervention_result(
            results_dir=results_dir,
            problem=row['problem'],
            intervention_raw_input_prompt=row['final_prompt'],
            intervention_raw_response=row['intervened_cot'],
            intervention_correct=1 if row['is_corrected'] else 0
        )

    if not intervention_df.empty:
        tested_problems = len(set(intervention_df['problem_id']))
        print_timestamped_message(f"Intervention testing complete: {tested_problems} problems with analyzable mistakes tested")
        
        correction_rates = intervention_df.groupby('condition')['is_corrected'].mean() * 100
        print_timestamped_message("Intervention testing complete. Correction rates:")
        print(correction_rates.to_string(float_format="%.2f%%"))

    return intervention_df


def run_activation_patching_experiment(model_interface, patching_setup, results_dir):
    """
    Runs activation patching to identify causally important model components.
    Now supports both toy examples and real intervention data.
    """
    print_timestamped_message("Starting activation patching experiment...")
    
    patching_model = model_interface.get_model_for_patching()
    if patching_model is None:
        print_timestamped_message("Model does not support patching. Skipping.")
        return None, None
        
    tokenizer = patching_model.tokenizer

    if patching_setup.get('use_intervention_data', False):
        return run_intervention_based_patching(patching_model, patching_setup, results_dir)
    else:
        return run_toy_example_patching(patching_model, patching_setup, results_dir)


def run_toy_example_patching(patching_model, patching_setup, results_dir):
    """Original toy example patching for simple cases."""
    tokenizer = patching_model.tokenizer
    
    # --- Tokenize clean and corrupted inputs ---
    clean_tokens = patching_model.to_tokens(patching_setup['clean_prompt'])
    corrupted_tokens = patching_model.to_tokens(patching_setup['corrupted_prompt'])
    
    # --- Get token IDs for answers ---
    correct_tokens = tokenizer.encode(patching_setup['correct_answer'], add_special_tokens=False)
    incorrect_tokens = tokenizer.encode(patching_setup['incorrect_answer'], add_special_tokens=False)
    
    print_timestamped_message(f"Correct answer '{patching_setup['correct_answer']}' tokenized as: {correct_tokens}")
    print_timestamped_message(f"Incorrect answer '{patching_setup['incorrect_answer']}' tokenized as: {incorrect_tokens}")
    
    if len(correct_tokens) != 1 or len(incorrect_tokens) != 1:
        print_timestamped_message("⚠️ Warning: Answers are not single tokens. Using first token of each.")
    
    correct_answer_token = correct_tokens[0]
    incorrect_answer_token = incorrect_tokens[0]

    # --- Debug: Check baseline logit differences ---
    clean_logits = patching_model.run_with_hooks(clean_tokens, fwd_hooks=[])
    corrupted_logits = patching_model.run_with_hooks(corrupted_tokens, fwd_hooks=[])
    
    clean_logit_diff = get_logit_diff(clean_logits, correct_answer_token, incorrect_answer_token)
    corrupted_logit_diff = get_logit_diff(corrupted_logits, correct_answer_token, incorrect_answer_token)
    
    print_timestamped_message(f"Clean logit diff: {clean_logit_diff:.4f}")
    print_timestamped_message(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")
    print_timestamped_message(f"Difference: {clean_logit_diff - corrupted_logit_diff:.4f}")
    
    if abs(clean_logit_diff - corrupted_logit_diff) < 1e-6:
        print_timestamped_message("⚠️ Warning: Clean and corrupted logit differences are nearly identical!")
        print_timestamped_message("This suggests the model doesn't distinguish between correct and incorrect answers.")

    # --- Run patching for specified components ---
    results = {}
    for component in patching_setup['patching_components']:
        print_timestamped_message(f"Patching component: {component}...")
        
        recovered_diffs = perform_activation_patching(
            patching_model,
            clean_tokens,
            corrupted_tokens,
            correct_answer_token,
            incorrect_answer_token,
            component
        )
        results[component] = recovered_diffs
    
    # Process and return results
    return process_patching_results(results, patching_model, patching_setup, results_dir)


def run_intervention_based_patching(patching_model, patching_setup, results_dir):
    """
    Runs activation patching using real intervention data.
    Uses problems where interventions successfully corrected mistakes.
    """
    intervention_df = patching_setup['intervention_df']
    max_problems = patching_setup.get('max_problems_to_patch', 5)
    
    # Filter for successful corrections that we can use for patching
    successful_corrections = intervention_df[intervention_df['is_corrected'] == True].copy()
    
    if successful_corrections.empty:
        print_timestamped_message("⚠️ No successful corrections found for activation patching.")
        return None, None
    
    # Limit the number of problems for computational efficiency
    if len(successful_corrections) > max_problems:
        successful_corrections = successful_corrections.head(max_problems)
    
    print_timestamped_message(f"Running activation patching on {len(successful_corrections)} successful corrections...")
    
    # We'll aggregate results across all problems
    all_results = {}
    
    for idx, row in successful_corrections.iterrows():
        problem_id = row['problem_id']
        print_timestamped_message(f"Processing problem {problem_id}...")
        
        # Create clean and corrupted prompts from the intervention data
        # Clean: Problem with correct reasoning leading to correct answer
        # Corrupted: Problem with incorrect reasoning leading to wrong answer
        
        problem_text = row['problem']
        correct_answer = row['ground_truth_answer']
        
        # Try to get the baseline incorrect answer for more realistic corrupted prompt
        baseline_solution = None
        incorrect_answer = None
        
        try:
            # Get baseline solution from database
            from database import get_experiment_results
            experiment_results = get_experiment_results(results_dir)
            for result in experiment_results:
                if result['problem'] == row['problem']:
                    baseline_solution = result['baseline_raw_response']
                    break
            
            if baseline_solution:
                from low_level import extract_boxed_answer
                incorrect_answer = extract_boxed_answer(baseline_solution)
        except:
            pass
        
        # Create more realistic prompts
        if incorrect_answer and incorrect_answer != str(correct_answer):
            clean_prompt = f"Solve this step by step:\n\n{problem_text}\n\nThe answer is {correct_answer}"
            corrupted_prompt = f"Solve this step by step:\n\n{problem_text}\n\nThe answer is {incorrect_answer}"
        else:
            # Fallback to generic approach
            clean_prompt = f"Solve this step by step:\n\n{problem_text}\n\nThe answer is {correct_answer}"
            corrupted_prompt = f"Solve this step by step:\n\n{problem_text}\n\nThe answer is 42"
        
        try:
            clean_tokens = patching_model.to_tokens(clean_prompt)
            corrupted_tokens = patching_model.to_tokens(corrupted_prompt)
            
            # For multi-token answers, we'll use a different approach
            # We'll look at the logit difference for the first token of the correct answer
            correct_tokens = patching_model.tokenizer.encode(str(correct_answer), add_special_tokens=False)
            
            if len(correct_tokens) == 0:
                print_timestamped_message(f"⚠️ Skipping problem {problem_id}: Could not tokenize answer")
                continue
                
            correct_answer_token = correct_tokens[0]
            
            # Use the incorrect answer we already extracted above
            if incorrect_answer and incorrect_answer != str(correct_answer):
                incorrect_tokens = patching_model.tokenizer.encode(str(incorrect_answer), add_special_tokens=False)
                if len(incorrect_tokens) > 0:
                    incorrect_answer_token = incorrect_tokens[0]
                    print_timestamped_message(f"  Using baseline incorrect answer '{incorrect_answer}' (token {incorrect_answer_token}) vs correct '{correct_answer}' (token {correct_answer_token})")
                else:
                    # Fallback to a different wrong answer
                    incorrect_answer_token = patching_model.tokenizer.encode("42", add_special_tokens=False)[0]
                    print_timestamped_message(f"  Fallback: Using '42' as incorrect token vs correct '{correct_answer}'")
            else:
                # Fallback if we can't extract or it matches correct answer
                incorrect_answer_token = patching_model.tokenizer.encode("42", add_special_tokens=False)[0]
                print_timestamped_message(f"  Using fallback '42' as incorrect token vs correct '{correct_answer}'")
            
            # Run patching for each component
            for component in patching_setup['patching_components']:
                if component not in all_results:
                    all_results[component] = []
                
                print_timestamped_message(f"  Patching {component} for problem {problem_id}...")
                
                recovered_diffs = perform_activation_patching(
                    patching_model,
                    clean_tokens,
                    corrupted_tokens,
                    correct_answer_token,
                    incorrect_answer_token,
                    component
                )
                
                all_results[component].append(recovered_diffs)
                
        except Exception as e:
            print_timestamped_message(f"⚠️ Error processing problem {problem_id}: {e}")
            continue
    
    if not all_results:
        print_timestamped_message("⚠️ No problems could be processed for activation patching.")
        return None, None
    
    # Average results across all problems
    final_results = {}
    for component, result_list in all_results.items():
        if result_list:
            # Find the minimum dimensions to handle different sequence lengths
            min_shape = result_list[0].shape
            for result in result_list[1:]:
                min_shape = tuple(min(min_shape[i], result.shape[i]) for i in range(len(min_shape)))
            
            # Truncate all results to the minimum shape and then stack
            truncated_results = []
            for result in result_list:
                if len(min_shape) == 3:  # For attention heads (layers, heads, positions)
                    truncated = result[:min_shape[0], :min_shape[1], :min_shape[2]]
                else:  # For other components (layers, positions)
                    truncated = result[:min_shape[0], :min_shape[1]]
                truncated_results.append(truncated)
            
            stacked_results = np.stack(truncated_results, axis=0)
            final_results[component] = np.mean(stacked_results, axis=0)
            print_timestamped_message(f"Averaged {len(result_list)} results for {component} (shape: {final_results[component].shape})")
    
    results = final_results

    # Continue with the rest of the processing (save, visualize, etc.)
    return process_patching_results(results, patching_model, patching_setup, results_dir)


def process_patching_results(results, patching_model, patching_setup, results_dir):
    """
    Common processing for both toy and intervention-based patching results.
    """
    # --- Save results to database and files ---
    print_timestamped_message("Saving activation patching results to database...")
    model_id = patching_model.tokenizer.name_or_path if hasattr(patching_model.tokenizer, 'name_or_path') else "unknown"
    
    for component_name, recovery_matrix in results.items():
        insert_activation_patching_result(
            results_dir=results_dir,
            model_id=model_id,
            patching_setup=patching_setup,
            component_name=component_name,
            recovery_matrix=recovery_matrix
        )
    
    output_file = os.path.join(results_dir, "activation_patching.json")
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json.dump({k: v.tolist() for k, v in results.items()}, f, indent=4)
    print_timestamped_message(f"Activation patching results saved to database and {output_file}")

    # Generate and save heatmaps for all components
    fig = None
    for component in patching_setup['patching_components']:
        heatmap_data = results[component]
        
        if component == "z" and len(heatmap_data.shape) == 3:
            # For attention heads, create a heatmap for each layer showing heads vs positions
            n_layers, n_heads, n_positions = heatmap_data.shape
            
            # Create a combined heatmap showing the maximum recovery across all positions for each head
            max_recovery_per_head = np.max(heatmap_data, axis=2)  # Shape: (layers, heads)
            
            component_fig = px.imshow(
                max_recovery_per_head,
                labels=dict(x="Head", y="Layer", color="Max Logit Diff Recovery"),
                x=[f"H{i}" for i in range(n_heads)],
                y=[f"L{i}" for i in range(n_layers)],
                title=f"Attention Head Patching: {component} (Max Recovery per Head)",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0
            )
            
            heatmap_file = os.path.join(results_dir, f"activation_patching_{component}_heads.html")
            component_fig.write_html(heatmap_file)
            print_timestamped_message(f"Attention head heatmap saved to {heatmap_file}")
            
            # Also create a detailed view for the best layer
            best_layer = np.unravel_index(np.argmax(max_recovery_per_head), max_recovery_per_head.shape)[0]
            detailed_fig = px.imshow(
                heatmap_data[best_layer],  # Shape: (heads, positions)
                labels=dict(x="Position", y="Head", color="Logit Diff Recovery"),
                x=[f"Tok {i}" for i in range(n_positions)],
                y=[f"H{i}" for i in range(n_heads)],
                title=f"Attention Head Patching: Layer {best_layer} Detail",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0
            )
            
            detailed_file = os.path.join(results_dir, f"activation_patching_{component}_layer{best_layer}_detail.html")
            detailed_fig.write_html(detailed_file)
            print_timestamped_message(f"Detailed attention head heatmap saved to {detailed_file}")
            
        else:
            # For 2D components, use the original approach
            component_fig = px.imshow(
                heatmap_data,
                labels=dict(x="Position", y="Layer", color="Logit Diff Recovery"),
                x=[f"Tok {i}" for i in range(heatmap_data.shape[1])],
                y=[f"L{i}" for i in range(heatmap_data.shape[0])],
                title=f"Activation Patching Recovery ({component})",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0
            )
            
            heatmap_file = os.path.join(results_dir, f"activation_patching_{component}.html")
            component_fig.write_html(heatmap_file)
            print_timestamped_message(f"Heatmap saved to {heatmap_file}")
        
        # Keep the first figure for return
        if fig is None:
            fig = component_fig
    
    # Save a readable summary of the heatmap data
    summary_file = os.path.join(results_dir, "activation_patching_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("ACTIVATION PATCHING RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for component, data in results.items():
            import numpy as np
            data_array = np.array(data)
            f.write(f"Component: {component}\n")
            f.write(f"Shape: {data_array.shape} (layers x positions)\n")
            f.write(f"Max Recovery: {np.max(data_array):.4f}\n")
            f.write(f"Mean Recovery: {np.mean(data_array):.4f}\n")
            f.write(f"Min Recovery: {np.min(data_array):.4f}\n")
            
            # Find top 5 positions with highest recovery
            flat_indices = np.argsort(data_array.flatten())[-5:][::-1]
            positions = [np.unravel_index(idx, data_array.shape) for idx in flat_indices]
            f.write("Top 5 positions (layer, position): recovery\n")
            for i, pos in enumerate(positions):
                recovery = data_array[pos]
                f.write(f"  {i+1}. Layer {pos[0]}, Position {pos[1]}: {recovery:.4f}\n")
            
            f.write("\nFull heatmap data (first 10 layers, first 10 positions):\n")
            display_data = data_array[:min(10, data_array.shape[0]), :min(10, data_array.shape[1])]
            f.write("     " + " ".join([f"Pos{i:2d}" for i in range(display_data.shape[1])]) + "\n")
            for layer in range(display_data.shape[0]):
                f.write(f"L{layer:2d}: " + " ".join([f"{display_data[layer, pos]:5.2f}" for pos in range(display_data.shape[1])]) + "\n")
            f.write("\n" + "-" * 50 + "\n\n")
    
    print_timestamped_message(f"Readable summary saved to {summary_file}")
        
    return results, fig


def run_steering_experiment(model_interface, baseline_df, intervention_df, intervention_phrases, results_dir, layer=21, max_validation_samples=None):
    """
    Runs steering vector experiment: creates vector from successful interventions and tests on validation set.
    
    Args:
        max_validation_samples: Maximum number of validation samples to test (None for no limit)
    """
    print_timestamped_message("Starting steering vector experiment...")
    
    # Only run for HuggingFace models
    if not hasattr(model_interface, 'get_model_for_patching'):
        print_timestamped_message("⚠️ Steering experiment skipped - model doesn't support TransformerLens")
        return None
    
    patching_model = model_interface.get_model_for_patching()
    if patching_model is None:
        print_timestamped_message("⚠️ Steering experiment skipped - could not get patching model")
        return None
    
    # 1. Calculate steering vector and get validation problems
    steering_vector, train_problems, validation_problems = calculate_steering_vector(
        model_interface, intervention_df, baseline_df, intervention_phrases, layer
    )
    
    if steering_vector is None or not validation_problems:
        print_timestamped_message("⚠️ Could not create steering vector or no validation problems available")
        return None
    
    # Apply validation sample limit if specified
    if max_validation_samples is not None and len(validation_problems) > max_validation_samples:
        validation_problems = validation_problems[:max_validation_samples]
        print_timestamped_message(f"Limited to {max_validation_samples} validation samples for faster testing")
    
    print_timestamped_message(f"Testing steering vector on {len(validation_problems)} validation problems...")
    
    # 2. Test steering vector on validation problems
    steering_results = []
    
    for problem_data in tqdm(validation_problems, desc="Testing Steering Vector"):
        # Apply steering vector at error location
        steered_solution, success = apply_steering_vector(
            model_interface, steering_vector, problem_data, layer, steering_strength=1.0
        )
        
        if success:
            # Evaluate the steered solution
            eval_result = evaluate_answer(
                problem_data['problem'], 
                problem_data['ground_truth_answer'], 
                steered_solution
            )
            
            steering_results.append({
                'problem_id': problem_data['problem_id'],
                'problem': problem_data['problem'],
                'ground_truth_answer': problem_data['ground_truth_answer'],
                'original_cot': problem_data['original_cot'],
                'error_line_content': problem_data['error_line_content'],
                'error_line_number': problem_data['error_line_number'],
                'steered_solution': steered_solution,
                'steered_answer': eval_result['generated_answer'],
                'is_corrected_by_steering': eval_result['is_correct'],
                'evaluation_method': eval_result['evaluation_method']
            })
        else:
            steering_results.append({
                'problem_id': problem_data['problem_id'],
                'problem': problem_data['problem'],
                'ground_truth_answer': problem_data['ground_truth_answer'],
                'original_cot': problem_data['original_cot'],
                'error_line_content': problem_data['error_line_content'],
                'error_line_number': problem_data['error_line_number'],
                'steered_solution': steered_solution,  # Contains error message
                'steered_answer': None,
                'is_corrected_by_steering': False,
                'evaluation_method': 'steering_failed'
            })
    
    # 3. Calculate and save results
    steering_df = pd.DataFrame(steering_results)
    
    if not steering_df.empty:
        success_rate = (steering_df['is_corrected_by_steering'].sum() / len(steering_df)) * 100
        print_timestamped_message(f"✅ Steering vector success rate: {success_rate:.1f}% ({steering_df['is_corrected_by_steering'].sum()}/{len(steering_df)})")
        
        # Save detailed results
        steering_file = os.path.join(results_dir, "steering_results.json")
        with open(steering_file, 'w') as f:
            json.dump({
                'layer': layer,
                'train_problems': train_problems,
                'validation_results': steering_results,
                'success_rate': success_rate,
                'total_validation_problems': len(validation_problems),
                'successful_corrections': int(steering_df['is_corrected_by_steering'].sum())
            }, f, indent=4, cls=NumpyEncoder)
        
        print_timestamped_message(f"Steering results saved to {steering_file}")
        
        # Save to database
        model_id = patching_model.tokenizer.name_or_path if hasattr(patching_model.tokenizer, 'name_or_path') else "unknown"
        insert_steering_result(
            results_dir=results_dir,
            model_id=model_id,
            layer=layer,
            train_problems_count=len(train_problems),
            validation_problems_count=len(validation_problems),
            success_rate=success_rate,
            successful_corrections=int(steering_df['is_corrected_by_steering'].sum()),
            steering_vector=steering_vector,
            validation_results=steering_results
        )
        
        # Save summary
        summary_file = os.path.join(results_dir, "steering_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("STEERING VECTOR EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Layer used: {layer}\n")
            f.write(f"Training problems: {len(train_problems)}\n")
            f.write(f"Validation problems: {len(validation_problems)}\n")
            f.write(f"Success rate: {success_rate:.1f}%\n")
            f.write(f"Successful corrections: {steering_df['is_corrected_by_steering'].sum()}\n\n")
            
            f.write("VALIDATION RESULTS:\n")
            f.write("-" * 30 + "\n")
            for _, row in steering_df.iterrows():
                f.write(f"Problem {row['problem_id']}: {'✓' if row['is_corrected_by_steering'] else '✗'}\n")
                f.write(f"  Error at line {row['error_line_number']}: {row['error_line_content'][:100]}...\n")
                if row['is_corrected_by_steering']:
                    f.write(f"  Steered answer: {row['steered_answer']}\n")
                f.write("\n")
        
        print_timestamped_message(f"Steering summary saved to {summary_file}")
        print_timestamped_message("Steering results saved to database.")
    
    return steering_df


# Removed: run_generalization_test and get_patching_results functions
# These functions implemented steering vector and activation patching functionality
