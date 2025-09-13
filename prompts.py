"""
Prompt creation functions for mathematical reasoning experiments.
"""


def create_baseline_prompt(problem_text):
    """
    Creates a baseline prompt for solving math problems from scratch.
    """
    return (
        f"Solve this math problem step-by-step. Be direct and focused. ALWAYS end with \\boxed{{answer}}.\n\n"
        f"Problem: {problem_text}\n\n"
        f"Solution: "
    )


def create_intervention_prompt(problem_text, original_cot, mistake_sentence, intervention_phrase):
    """
    Creates an intervention prompt that clearly informs the model this is a continuation task.
    The model is explicitly told that the reasoning was cut off and needs to be continued.
    
    Args:
        problem_text: The original math problem
        original_cot: The original chain of thought with the mistake
        mistake_sentence: The sentence containing the mistake
        intervention_phrase: The corrective phrase to inject
    
    Returns:
        The complete intervention prompt
    """
    # Find where the mistake occurs and truncate after it
    mistake_index = original_cot.find(mistake_sentence)
    if mistake_index == -1:
        raise ValueError(f"Mistake sentence not found in CoT: {mistake_sentence}")
    
    mistake_end_index = mistake_index + len(mistake_sentence)
    truncated_cot = original_cot[:mistake_end_index]
    
    # Create a neutral intervention prompt - NO bias about errors!
    # The goal is to test if the intervention phrase alone causes rethinking
    prompt = (
        f"You are solving a math problem step by step. "
        f"Continue the reasoning from where it left off. "
        f"ALWAYS end with \\boxed{{answer}}.\n\n"
        f"Problem: {problem_text}\n\n"
        f"Solution so far: {truncated_cot}{intervention_phrase}\n\n"
    )
    
    return prompt


