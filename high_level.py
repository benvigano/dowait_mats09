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

from database import insert_baseline_result, insert_intervention_result

from cache import (
    print_timestamped_message, get_from_generation_cache, 
    get_from_generalization_cache, save_to_generalization_cache
)
from low_level import (
    solve_problem_baseline, solve_problem_with_intervention,
    extract_boxed_answer, is_correct, evaluate_answer
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

