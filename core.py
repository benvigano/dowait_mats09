"""
Core orchestrator - imports from specialized modules and provides legacy compatibility.
"""

# --- Legacy imports for backward compatibility ---
from cache import (
    print_timestamped_message, get_timestamp_in_rome,
    save_to_generation_cache as _save_to_cache,
    get_from_generation_cache as _get_from_cache,
    save_to_error_cache_by_key as _save_to_error_cache_by_key,
    get_from_error_cache_by_key as _get_from_error_cache_by_key,
    _get_error_cache_key,
    clear_generation_cache
)

from low_level import (
    select_diverse_problems,
    load_model_and_tokenizer,
    generate_with_model,
    solve_problem_baseline,
    solve_problem_with_intervention,
    extract_boxed_answer,
    is_correct,
    evaluate_answer
)

from prompts import (
    create_baseline_prompt,
    create_intervention_prompt
)

from high_level import (
    run_baseline_experiment,
    run_insertion_test
)

from database import print_database_summary, get_experiment_results

print_timestamped_message("Core module loaded successfully. All functionality imported from specialized modules.")
print_timestamped_message("⚠️ Note: Configuration constants moved to notebook.py for better modularity.")