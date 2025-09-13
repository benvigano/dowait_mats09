#!/usr/bin/env python3
"""
Dataset-specific tensor error investigation.
Load the actual MATH dataset and test problematic entries.
"""

import os
import sys
from datasets import load_dataset
from core import (
    load_model_and_tokenizer, solve_problem_baseline, 
    print_timestamped_message, PROD_SAMPLE_SIZE
)

def find_problematic_entries(dataset, model):
    """Test each dataset entry to find which ones cause tensor errors."""
    print_timestamped_message(f"🔍 Testing {len(dataset)} dataset entries for tensor errors...")
    
    problematic_indices = []
    successful_indices = []
    
    for i, example in enumerate(dataset):
        problem = example['problem']
        print_timestamped_message(f"   Testing entry {i+1}/{len(dataset)}: {problem[:50]}...")
        
        try:
            full_solution, raw_prompt = solve_problem_baseline(model, problem)
            print_timestamped_message(f"      ✅ SUCCESS: {len(full_solution)} chars")
            successful_indices.append(i)
            
        except Exception as e:
            error_msg = str(e)
            print_timestamped_message(f"      ❌ FAILED: {error_msg}")
            
            if "dimension 3" in error_msg:
                print_timestamped_message(f"      🎯 TENSOR ERROR FOUND!")
                problematic_indices.append(i)
                
                # Log detailed info about this problematic entry
                print_timestamped_message(f"      🔍 PROBLEMATIC ENTRY ANALYSIS:")
                print_timestamped_message(f"         Index: {i}")
                print_timestamped_message(f"         Problem: {problem}")
                print_timestamped_message(f"         Problem length: {len(problem)} chars")
                print_timestamped_message(f"         Contains ceiling: {'lceil' in problem}")
                print_timestamped_message(f"         Contains floor: {'lfloor' in problem}")
                print_timestamped_message(f"         Contains fractions: {'frac' in problem}")
                print_timestamped_message(f"         Contains dfrac: {'dfrac' in problem}")
                print_timestamped_message(f"         Non-ASCII chars: {any(ord(c) > 127 for c in problem)}")
                
                # Show problem in hex to see any weird characters
                problem_hex = problem.encode('utf-8').hex()
                print_timestamped_message(f"         Problem hex (first 200 chars): {problem_hex[:400]}")
                
                print_timestamped_message(f"         Answer: {example.get('solution', 'N/A')}")
                print_timestamped_message("")
                
                # Test if it's position-dependent by trying the same problem again
                try:
                    print_timestamped_message(f"      🔄 RETRY TEST:")
                    full_solution, raw_prompt = solve_problem_baseline(model, problem)
                    print_timestamped_message(f"         ✅ RETRY SUCCESS - Position dependent error!")
                except Exception as retry_e:
                    print_timestamped_message(f"         ❌ RETRY FAILED - Consistent error: {str(retry_e)}")
    
    return problematic_indices, successful_indices

def analyze_patterns(dataset, problematic_indices, successful_indices):
    """Analyze patterns in problematic vs successful entries."""
    print_timestamped_message("🔬 PATTERN ANALYSIS")
    
    if not problematic_indices:
        print_timestamped_message("   No problematic entries found.")
        return
    
    print_timestamped_message(f"   Problematic entries: {len(problematic_indices)}")
    print_timestamped_message(f"   Successful entries: {len(successful_indices)}")
    
    # Analyze problematic entries
    prob_problems = [dataset[i]['problem'] for i in problematic_indices]
    succ_problems = [dataset[i]['problem'] for i in successful_indices[:len(prob_problems)]]  # Same sample size
    
    # Length analysis
    prob_lengths = [len(p) for p in prob_problems]
    succ_lengths = [len(p) for p in succ_problems]
    
    print_timestamped_message(f"   Average length - Problematic: {sum(prob_lengths)/len(prob_lengths):.1f}")
    print_timestamped_message(f"   Average length - Successful: {sum(succ_lengths)/len(succ_lengths):.1f}")
    
    # Symbol analysis
    prob_ceil_count = sum('lceil' in p for p in prob_problems)
    succ_ceil_count = sum('lceil' in p for p in succ_problems)
    
    prob_frac_count = sum('frac' in p for p in prob_problems)
    succ_frac_count = sum('frac' in p for p in succ_problems)
    
    print_timestamped_message(f"   Ceiling functions - Problematic: {prob_ceil_count}/{len(prob_problems)}")
    print_timestamped_message(f"   Ceiling functions - Successful: {succ_ceil_count}/{len(succ_problems)}")
    
    print_timestamped_message(f"   Fractions - Problematic: {prob_frac_count}/{len(prob_problems)}")
    print_timestamped_message(f"   Fractions - Successful: {succ_frac_count}/{len(succ_problems)}")

def main():
    print_timestamped_message("🎯 Dataset-Specific Tensor Error Investigation")
    
    # Load dataset
    print_timestamped_message("📊 Loading MATH dataset...")
    full_dataset = load_dataset("EleutherAI/hendrycks_math", "algebra", split='test')
    dataset = full_dataset.select(range(PROD_SAMPLE_SIZE))  # Same as experiment
    print_timestamped_message(f"   Loaded {len(dataset)} problems (matching experiment size)")
    
    # Load model
    print_timestamped_message("🔧 Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Find problematic entries
    problematic_indices, successful_indices = find_problematic_entries(dataset, model)
    
    # Analyze patterns
    analyze_patterns(dataset, problematic_indices, successful_indices)
    
    print_timestamped_message("")
    print_timestamped_message("🎯 SUMMARY:")
    if problematic_indices:
        print_timestamped_message(f"   ✅ REPRODUCED tensor errors in {len(problematic_indices)} entries!")
        print_timestamped_message(f"   📍 Problematic indices: {problematic_indices}")
        print_timestamped_message("   🔧 Next: Analyze the specific problematic entries for common patterns")
    else:
        print_timestamped_message("   ❓ No tensor errors reproduced. Possible causes:")
        print_timestamped_message("      - Model state dependent on batch processing")
        print_timestamped_message("      - Memory/GPU state issues")
        print_timestamped_message("      - Caching system interference")

if __name__ == "__main__":
    main()

