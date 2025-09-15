"""
Low-level utilities: model loading, generation, evaluation, prompts.
"""

import torch
import re
import traceback
import time
import os
import json
import anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from functools import partial

from cache import (
    get_from_generation_cache, save_to_generation_cache, 
    get_from_error_cache_by_key, save_to_error_cache_by_key,
    _get_error_cache_key, print_timestamped_message
)

# Note: Configuration constants moved to notebook.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Problem Selection ---

def select_diverse_problems(total_problems_wanted=30, debug_mode=False, seed=42):
    """
    Select problems from all MATH subjects with heavy bias toward harder problems.
    Heavily weights Level 4-5 problems to maximize errors for correction experiments.
    Selection is deterministic for reproducible experiments.
    
    Args:
        total_problems_wanted: Total number of problems to select
        debug_mode: If True, use a small deterministic sample
        seed: Random seed for deterministic selection
        
    Returns:
        List of problem dictionaries with added metadata (skewed toward hard problems)
    """
    from datasets import load_dataset
    import random
    
    # Set seed for deterministic selection
    random.seed(seed)
    
    if debug_mode:
        total_problems_wanted = 10
        print_timestamped_message(f"🔧 DEBUG MODE: Selecting {total_problems_wanted} problems (seed={seed})")
    else:
        print_timestamped_message(f"📊 Selecting {total_problems_wanted} problems heavily biased toward hard problems (seed={seed})")
    
    subjects = [
        'precalculus',           # Hardest - prioritize
        'intermediate_algebra',  # Very hard
        'counting_and_probability',  # Hard
        'number_theory',         # Moderately hard
        'geometry',              # Moderate
        'algebra',               # Easier
        'prealgebra'             # Easiest - minimal
    ]
    
    # Level weights: heavily favor harder problems to get more errors to correct
    level_weights = {
        'Level 1': 1,    # Easiest - minimal weight
        'Level 2': 2, 
        'Level 3': 6,    # Start ramping up
        'Level 4': 12,   # Heavy weight on hard problems
        'Level 5': 20    # Maximum weight on hardest problems
    }
    
    selected_problems = []
    problems_per_subject = max(1, total_problems_wanted // len(subjects))
    
    for subject in subjects:
        print_timestamped_message(f"  Loading {subject}...")
        dataset = load_dataset('EleutherAI/hendrycks_math', subject, split='test')
        
        # Group problems by level 
        problems_by_level = {}
        for i, problem in enumerate(dataset):
            level = problem['level']
            if level not in problems_by_level:
                problems_by_level[level] = []
            
            # Add metadata
            problem_with_meta = {
                'problem': problem['problem'],
                'solution': problem['solution'],
                'level': level,
                'subject': subject,
                'dataset_index': i
            }
            problems_by_level[level].append(problem_with_meta)
        
        # Select problems weighted by difficulty
        subject_problems = []
        for level, weight in level_weights.items():
            if level in problems_by_level:
                level_problems = problems_by_level[level]
                # More problems from harder levels
                num_from_level = max(1, (weight * problems_per_subject) // sum(level_weights.values()))
                num_from_level = min(num_from_level, len(level_problems))
                
                # Deterministic selection (seeded random for reproducibility)
                if num_from_level >= len(level_problems):
                    selected = level_problems
                else:
                    selected = random.sample(level_problems, num_from_level)
                    
                subject_problems.extend(selected)
        
        selected_problems.extend(subject_problems)
        print_timestamped_message(f"    Selected {len(subject_problems)} problems from {subject}")
    
    # Shuffle for random order (deterministic due to seed)
    random.shuffle(selected_problems)
    
    # Trim to exact count
    selected_problems = selected_problems[:total_problems_wanted]
    
    print_timestamped_message(f"✅ Final selection: {len(selected_problems)} problems")
    
    # Show distribution
    level_counts = {}
    subject_counts = {}
    for p in selected_problems:
        level_counts[p['level']] = level_counts.get(p['level'], 0) + 1
        subject_counts[p['subject']] = subject_counts.get(p['subject'], 0) + 1
    
    print_timestamped_message(f"  Level distribution: {level_counts}")
    print_timestamped_message(f"  Subject distribution: {subject_counts}")
    
    return selected_problems

# --- Model Loading ---

def load_model_and_tokenizer(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """
    Loads the HuggingFace model and tokenizer for generation.
    Activation patching and steering disabled for stability.
    """
    print_timestamped_message(f"Loading HuggingFace model '{model_id}'...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print_timestamped_message("HuggingFace model loaded successfully.")
    print_timestamped_message(f"Model device: {next(model.parameters()).device}")
    
    return model, tokenizer

# --- Text Generation ---

def generate_with_model(model, prompt, max_new_tokens, temperature=0.2, top_p=0.75, do_sample=True):
    """
    Pure LLM inference wrapper with caching using HuggingFace.
    This function has no knowledge of problem types or prompt structures.
    
    Args:
        model: HuggingFace AutoModelForCausalLM model
        prompt: The complete prompt to send to the model
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter  
        do_sample: Whether to use sampling
    
    Returns:
        Generated text (only the new tokens, not including the prompt)
    """
    # Note: Caching now handled by ModelInterface implementations
    # This function is deprecated - use models.py instead
    
    # Get tokenizer from model (if available) or use global tokenizer
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        # Fallback: load tokenizer if not attached to model
        from transformers import AutoTokenizer
        # This fallback should not be used - models should include their own tokenizers
        raise ValueError("Model does not have an attached tokenizer and this function is deprecated")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with HuggingFace
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Note: Caching now handled by ModelInterface implementations
    # This function is deprecated
    
    return generated_text

# --- Problem Solving Functions ---

def solve_problem_baseline(model, problem_text):
    """
    Solves a math problem using the baseline approach.
    
    Returns:
        tuple: (generated_solution, full_prompt)
    """
    from prompts import create_baseline_prompt
    
    prompt = create_baseline_prompt(problem_text)
    
    # Use new model interface
    if hasattr(model, 'generate'):
        solution = model.generate(prompt)
    else:
        # Legacy fallback
        solution = generate_with_model(model, prompt)
    
    return solution, prompt

def solve_problem_with_intervention(model, problem_text, original_cot, mistake_sentence, intervention_phrase):
    """
    Solves a math problem using intervention at the point of error.
    
    Returns:
        tuple: (generated_solution, full_prompt)
    """
    from prompts import create_intervention_prompt
    
    prompt = create_intervention_prompt(problem_text, original_cot, mistake_sentence, intervention_phrase)
    
    # Use new model interface
    if hasattr(model, 'generate'):
        solution = model.generate(prompt)
    else:
        # Legacy fallback
        solution = generate_with_model(model, prompt)
    
    return solution, prompt


# --- Evaluation Functions ---

def extract_boxed_answer(text: str):
    """
    Extracts the content from the last \\boxed{...} block in a string.
    Uses brace counting to handle nested braces correctly (e.g., \\frac{3}{4}).
    """
    if not isinstance(text, str):
        return None
        
    # Find all \boxed{ patterns
    pattern = r'\\boxed\{'
    matches = []
    
    for match in re.finditer(pattern, text):
        start_pos = match.end()  # Position after \boxed{
        brace_count = 1
        pos = start_pos
        
        # Count braces to find the matching closing brace
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:  # Found matching closing brace
            content = text[start_pos:pos-1]
            matches.append(content)
    
    return matches[-1] if matches else None

def is_correct(ground_truth_answer, generated_answer):
    """
    Compares two answers for equivalence, handling multiple numbers, and basic LaTeX.
    This is a much more robust version.
    """
    if ground_truth_answer is None or generated_answer is None:
        return False

    # Normalize by removing ALL whitespace and converting to lowercase
    gt_norm = str(ground_truth_answer).replace(" ", "").replace("\t", "").replace("\n", "").strip().lower()
    gen_norm = str(generated_answer).replace(" ", "").replace("\t", "").replace("\n", "").strip().lower()

    # Direct comparison for non-numerical answers
    if gt_norm == gen_norm:
        return True

    # Numerical comparison logic
    # This regex handles integers, decimals, complex numbers, and basic fractions like \frac{a}{b}
    num_regex = r"-?\\d*\\.?\\d+i?|\\frac{\s*-?\\d+\s*}{\s*-?\\d+\s*}"
    
    gt_nums = re.findall(num_regex, gt_norm)
    gen_nums = re.findall(num_regex, gen_norm)

    # If there's a different number of numerical parts, they can't be equal
    if len(gt_nums) != len(gen_nums):
        return False
    
    # If no numbers found after all, and strings didn't match, they are not equal
    if not gt_nums:
        return False

    # Compare all numerical parts
    # For now, we'll do a direct string comparison of the extracted parts.
    # This correctly handles "4, 6, 14, 15" vs "4, 6, 14, 19"
    return gt_nums == gen_nums

def evaluate_answer(problem, ground_truth_answer, generated_solution):
    """
    Unified evaluation function that:
    1. Extracts the generated answer from the solution
    2. First tries algorithmic matching 
    3. Falls back to Anthropic evaluation if no match (also finds error line for incorrect answers)
    
    Returns:
        dict: {
            'is_correct': bool,
            'generated_answer': str or None,
            'evaluation_method': str,  # 'simple_match', 'anthropic_correct', 'anthropic_incorrect', 'anthropic_failure', 'extraction_failed'
            'evaluation_result': str,  # 'correct', 'incorrect', 'failure'
            'error_line_number': int or None,  # Line number where first error occurs (only for incorrect)
            'error_line_content': str or None,  # Content of the error line (only for incorrect)
            'mistake_sentence_usable': bool    # Whether error location is usable for intervention
        }
    """
    # Step 1: Extract the generated answer
    generated_answer = extract_boxed_answer(generated_solution)
    
    # Step 2: Try algorithmic matching first (only if we have an answer)
    if generated_answer is not None and is_correct(generated_answer, ground_truth_answer):
        return {
            'is_correct': True,
            'generated_answer': generated_answer,
            'evaluation_method': 'simple_match',
            'evaluation_result': 'correct',
            'error_line_number': None,
            'error_line_content': None,
            'mistake_sentence_usable': False
        }
    
    # Step 3: Fall back to Anthropic evaluation with error line detection
    # Call Anthropic even when answer extraction failed - it can still find error lines
    anthropic_result = _evaluate_with_anthropic_and_find_errors(problem, ground_truth_answer, generated_solution)
    
    return {
        'is_correct': anthropic_result['is_correct'],
        'generated_answer': generated_answer,
        'evaluation_method': f"anthropic_{anthropic_result['evaluation_result']}",
        'evaluation_result': anthropic_result['evaluation_result'],
        'error_line_number': anthropic_result.get('error_line_number'),
        'error_line_content': anthropic_result.get('error_line_content'),
        'mistake_sentence_usable': anthropic_result.get('mistake_sentence_usable', False)
    }

def _evaluate_with_anthropic_and_find_errors(problem, ground_truth_answer, full_solution):
    """
    Internal function that uses Anthropic to evaluate when algorithmic matching fails.
    Also finds the specific error line for incorrect solutions.
    
    Returns:
        dict: {
            'is_correct': bool,
            'evaluation_result': str,  # 'correct', 'incorrect', 'failure'
            'error_line_number': int or None,
            'error_line_content': str or None,
            'mistake_sentence_usable': bool
        }
    """
    # Check cache first
    cache_key = _get_error_cache_key(problem, full_solution + "_unified_eval")
    cached_result = get_from_error_cache_by_key(cache_key)
    if cached_result is not None:
        try:
            return json.loads(cached_result)
        except (json.JSONDecodeError, KeyError):
            # Invalid cache entry, continue with fresh analysis
            pass
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        result = {
            'is_correct': False,
            'evaluation_result': 'failure',
            'error_line_number': None,
            'error_line_content': "No API key available",
            'mistake_sentence_usable': False
        }
        save_to_error_cache_by_key(cache_key, problem, full_solution, json.dumps(result), "no_api_key")
        return result
    
    # Prepare the solution with line numbers
    lines = [line.strip() for line in full_solution.split('\n') if line.strip()]
    if not lines:
        result = {
            'is_correct': False,
            'evaluation_result': 'failure',
            'error_line_number': None,
            'error_line_content': "Empty solution",
            'mistake_sentence_usable': False
        }
        save_to_error_cache_by_key(cache_key, problem, full_solution, json.dumps(result))
        return result
    
    # Create numbered lines display
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        numbered_lines.append(f"{i:3d}: {line}")
    lines_display = "\n".join(numbered_lines)
    
    # Define the tool schema for structured output
    evaluation_tool = {
        "name": "evaluate_solution",
        "description": "Evaluates a mathematical solution and identifies errors if present",
        "input_schema": {
            "type": "object",
            "properties": {
                "is_correct": {
                    "type": "boolean",
                    "description": "True if the solution is mathematically correct (even if format differs), False if there are mathematical errors or failures"
                },
                "evaluation_result": {
                    "type": "string",
                    "enum": ["correct", "incorrect", "failure"],
                    "description": "Overall evaluation: 'correct' if mathematically sound, 'incorrect' if errors, 'failure' if incomplete/uninterpretable"
                },
                "error_line_number": {
                    "type": "integer",
                    "description": "Line number (1-based) where the first mathematical error occurs. Only provide if evaluation_result is 'incorrect'.",
                    "minimum": 1,
                    "maximum": len(lines)
                }
            },
            "required": ["is_correct", "evaluation_result"]
        }
    }
    
    prompt = f"""You are evaluating a math problem solution. The response didn't match our automated checker, but it might still be correct.

PROBLEM:
{problem}

CORRECT ANSWER:
{ground_truth_answer}

GENERATED RESPONSE (with line numbers):
{lines_display}

TASK:
1. Determine if the solution is mathematically correct (even if format differs)
2. If incorrect, identify the line number where the FIRST mathematical error occurs
3. Categorize as: 'correct', 'incorrect', or 'failure'

Look for:
- Arithmetic mistakes
- Incorrect formulas or algebraic manipulations
- Wrong problem interpretation
- Logical errors in reasoning
- Sign errors or computational mistakes
- Conceptual misunderstandings

Use the evaluate_solution tool to provide your analysis."""
    
    try:
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            tools=[evaluation_tool],
            tool_choice={"type": "tool", "name": "evaluate_solution"},
            messages=[{"role": "user", "content": prompt}]
        )
        time.sleep(2)  # Rate limiting
        
        # Parse the structured response
        result = {
            'is_correct': False,
            'evaluation_result': 'failure',
            'error_line_number': None,
            'error_line_content': None,
            'mistake_sentence_usable': False
        }
        
        if message.content:
            for content_block in message.content:
                if content_block.type == "tool_use" and content_block.name == "evaluate_solution":
                    tool_input = content_block.input
                    result['is_correct'] = tool_input.get('is_correct', False)
                    result['evaluation_result'] = tool_input.get('evaluation_result', 'failure')
                    
                    # If incorrect, extract error line information
                    if result['evaluation_result'] == 'incorrect':
                        error_line_num = tool_input.get('error_line_number')
                        if error_line_num and 1 <= error_line_num <= len(lines):
                            result['error_line_number'] = error_line_num
                            result['error_line_content'] = lines[error_line_num - 1]
                            result['mistake_sentence_usable'] = True
                    break
        
        # Cache the result
        save_to_error_cache_by_key(cache_key, problem, full_solution, json.dumps(result))
        return result
        
    except Exception as e:
        print_timestamped_message(f"Error in Anthropic evaluation: {e}")
        if 'rate_limit' in str(e).lower():
            print_timestamped_message("Rate limit hit. Waiting 30 seconds...")
            time.sleep(30)
            # One retry
            try:
                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    tools=[evaluation_tool],
                    tool_choice={"type": "tool", "name": "evaluate_solution"},
                    messages=[{"role": "user", "content": prompt}]
                )
                time.sleep(3)
                
                result = {
                    'is_correct': False,
                    'evaluation_result': 'failure',
                    'error_line_number': None,
                    'error_line_content': None,
                    'mistake_sentence_usable': False
                }
                
                if message.content:
                    for content_block in message.content:
                        if content_block.type == "tool_use" and content_block.name == "evaluate_solution":
                            tool_input = content_block.input
                            result['is_correct'] = tool_input.get('is_correct', False)
                            result['evaluation_result'] = tool_input.get('evaluation_result', 'failure')
                            
                            if result['evaluation_result'] == 'incorrect':
                                error_line_num = tool_input.get('error_line_number')
                                if error_line_num and 1 <= error_line_num <= len(lines):
                                    result['error_line_number'] = error_line_num
                                    result['error_line_content'] = lines[error_line_num - 1]
                                    result['mistake_sentence_usable'] = True
                            break
                
                save_to_error_cache_by_key(cache_key, problem, full_solution, json.dumps(result))
                return result
                
            except Exception as retry_error:
                print_timestamped_message(f"Retry failed: {retry_error}")
        
        error_result = {
            'is_correct': False,
            'evaluation_result': 'failure',
            'error_line_number': None,
            'error_line_content': f"Analysis failed: {str(e)}",
            'mistake_sentence_usable': False
        }
        # Don't save failed analyses to cache
        return error_result
