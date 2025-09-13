#!/usr/bin/env python3
"""
Context-Dependent Isolation Testing
Test under exact experimental conditions to reproduce tensor dimension errors.
"""

import os
import sys
import pandas as pd
from core import (
    load_model_and_tokenizer, solve_problem_baseline, 
    print_timestamped_message, evaluate_answer
)

def load_test_problems():
    """Load the exact problematic problems from MATH dataset."""
    try:
        # Try to load from existing results to get the exact problems
        from database import get_experiment_results
        results_dirs = [d for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))]
        if results_dirs:
            latest_dir = os.path.join("results", max(results_dirs))
            try:
                df = get_experiment_results(latest_dir)
                if not df.empty:
                    print_timestamped_message(f"📊 Loaded {len(df)} problems from existing results")
                    return df['problem'].tolist()[:20]  # Test first 20
            except:
                pass
    except:
        pass
    
    # Fallback: specific problematic problems mentioned in error logs
    return [
        "Find $x$ such that $\\lceil x \\rceil + x = \\dfrac{23}{7}$. Express $x$ as a common fraction.",
        "Mr. Madoff invests 1000 dollars in a fund that compounds annually at a constant rate of $13\\%$. Mr. Roth invests 2000 dollars in a fund that compounds annually at a constant rate of $10\\%$. After how many whole years will Mr. Madoff's account have more money than Mr. Roth's account?",
        "Four distinct integers $a$, $b$, $c$ and $d$ have the property that when added in pairs, the sums $16$, $19$, $20$, $21$, $22$, and $25$ are obtained. Find the four integers.",
        "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?",
        "If $\\cos \\theta = \\frac{1}{3}$, what is $\\cos 2\\theta$?",
    ]

def test_sequential_generation(model, problems):
    """Test problems sequentially like in the main experiment."""
    print_timestamped_message("🔄 Testing Sequential Generation (Main Experiment Simulation)")
    
    successful = 0
    failed = 0
    
    for i, problem in enumerate(problems):
        print_timestamped_message(f"📝 Problem {i+1}/{len(problems)}")
        print_timestamped_message(f"   Problem: {problem[:80]}...")
        
        try:
            # This exactly mimics the main experiment workflow
            full_solution, raw_prompt = solve_problem_baseline(model, problem)
            print_timestamped_message(f"   ✅ SUCCESS: Generated {len(full_solution)} chars")
            successful += 1
            
            # Also test evaluation (this sometimes triggers errors too)
            try:
                # Use a dummy ground truth for testing
                eval_result = evaluate_answer(problem, "dummy_answer", full_solution)
                print_timestamped_message(f"   ✅ Evaluation also succeeded")
            except Exception as eval_e:
                print_timestamped_message(f"   ⚠️  Evaluation failed: {str(eval_e)}")
                
        except Exception as e:
            print_timestamped_message(f"   ❌ FAILED: {str(e)}")
            if "dimension 3" in str(e):
                print_timestamped_message(f"   🎯 TENSOR ERROR REPRODUCED!")
                # Log additional debug info
                print_timestamped_message(f"   🔍 Problem text length: {len(problem)}")
                print_timestamped_message(f"   🔍 Contains ceiling: {'lceil' in problem}")
                print_timestamped_message(f"   🔍 Contains floor: {'lfloor' in problem}")
                print_timestamped_message(f"   🔍 Contains fractions: {'frac' in problem}")
            failed += 1
        
        print_timestamped_message("")
    
    return successful, failed

def test_memory_accumulation(model, problem):
    """Test if memory accumulation causes the error."""
    print_timestamped_message("🧠 Testing Memory Accumulation")
    print_timestamped_message(f"   Repeating same problem multiple times...")
    
    for i in range(10):
        try:
            full_solution, raw_prompt = solve_problem_baseline(model, problem)
            print_timestamped_message(f"   Iteration {i+1}: ✅ SUCCESS ({len(full_solution)} chars)")
        except Exception as e:
            print_timestamped_message(f"   Iteration {i+1}: ❌ FAILED - {str(e)}")
            if "dimension 3" in str(e):
                print_timestamped_message(f"   🎯 TENSOR ERROR AFTER {i+1} ITERATIONS!")
                return i+1
    
    print_timestamped_message("   🎉 All 10 iterations successful")
    return None

def test_with_max_tokens(model, problem):
    """Test with different max_tokens settings."""
    print_timestamped_message("🔢 Testing Different Max Tokens")
    
    token_limits = [100, 500, 1024, 2048, 4096]
    
    for limit in token_limits:
        try:
            from low_level import generate_with_model, create_baseline_prompt
            prompt = create_baseline_prompt(problem)
            result = generate_with_model(model, prompt, max_new_tokens=limit)
            print_timestamped_message(f"   max_tokens={limit}: ✅ SUCCESS ({len(result)} chars)")
        except Exception as e:
            print_timestamped_message(f"   max_tokens={limit}: ❌ FAILED - {str(e)}")
            if "dimension 3" in str(e):
                print_timestamped_message(f"   🎯 TENSOR ERROR AT {limit} TOKENS!")

def main():
    print_timestamped_message("🎯 Starting Context-Dependent Isolation Testing")
    
    # Load model
    print_timestamped_message("🔧 Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Load test problems
    problems = load_test_problems()
    print_timestamped_message(f"📋 Loaded {len(problems)} test problems")
    
    # Test 1: Sequential generation (main experiment simulation)
    successful, failed = test_sequential_generation(model, problems)
    
    print_timestamped_message("=" * 60)
    print_timestamped_message(f"📊 SEQUENTIAL TEST RESULTS: {successful} passed, {failed} failed")
    
    if failed > 0:
        print_timestamped_message("🎯 TENSOR ERROR REPRODUCED IN SEQUENTIAL TESTING!")
    else:
        print_timestamped_message("🤔 No errors in sequential testing. Testing other conditions...")
        
        # Test 2: Memory accumulation
        test_problem = problems[0]  # Use first problem for memory test
        failure_iteration = test_memory_accumulation(model, test_problem)
        
        if failure_iteration:
            print_timestamped_message(f"🎯 MEMORY ACCUMULATION ERROR AFTER {failure_iteration} ITERATIONS!")
        else:
            # Test 3: Different token limits
            test_with_max_tokens(model, test_problem)
    
    print_timestamped_message("")
    print_timestamped_message("🔍 ANALYSIS COMPLETE")
    print_timestamped_message("   If no errors reproduced, the issue may be:")
    print_timestamped_message("   1. Specific dataset problems not tested")
    print_timestamped_message("   2. Model state dependent on previous operations")
    print_timestamped_message("   3. GPU memory fragmentation")
    print_timestamped_message("   4. Caching system interference")

if __name__ == "__main__":
    main()

