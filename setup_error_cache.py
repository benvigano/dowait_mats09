#!/usr/bin/env python3
"""
One-time error detection cache setup script to pre-populate from existing Excel files.
Run this once to extract all previous error detection results into the cache.

Usage: uv run setup_error_cache.py
"""

import core
import pandas as pd
import os

def prepopulate_error_cache_from_excel_files():
    """Pre-populate the error detection cache with existing error analysis results."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        core.print_timestamped_message("No results directory found, skipping error cache pre-population")
        return
    
    total_added = 0
    excel_files = [f for f in os.listdir(results_dir) if f.endswith('.xlsx')]
    
    core.print_timestamped_message(f"Pre-populating error detection cache from {len(excel_files)} Excel files...")
    
    for excel_file in excel_files:
        excel_path = os.path.join(results_dir, excel_file)
        try:
            # Try to read Error_Analysis sheet
            xl_file = pd.ExcelFile(excel_path)
            if 'Error_Analysis' in xl_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name='Error_Analysis')
                
                # Extract valid entries (non-null problem, solution, and error sentence)
                valid_entries = df.dropna(subset=['problem', 'full_generated_solution', 'error_sentence'])
                
                for _, row in valid_entries.iterrows():
                    problem_text = row['problem']
                    incorrect_cot = row['full_generated_solution']
                    error_sentence = row['error_sentence']
                    
                    # Skip placeholder entries
                    if 'ANTHROPIC_API_KEY not set' in str(error_sentence):
                        continue
                    
                    # Check if this is already cached
                    cached_result = core._get_from_error_cache(problem_text, incorrect_cot)
                    if cached_result is None:
                        # Add to cache (without printing cache hit message)
                        cache_key = core._get_error_cache_key(problem_text, incorrect_cot)
                        cache = core._load_error_detection_cache()
                        cache[cache_key] = {
                            'problem_text': problem_text,
                            'incorrect_cot': incorrect_cot,
                            'error_sentence': error_sentence,
                            'timestamp': f"pre-populated from {excel_file}",
                            'api_model': 'claude-3-5-sonnet-20240620'
                        }
                        total_added += 1
                
                core.print_timestamped_message(f"Processed {excel_file}: {len(valid_entries)} error analysis entries")
                
        except Exception as e:
            core.print_timestamped_message(f"Could not process {excel_file}: {e}")
    
    if total_added > 0:
        # Save the updated cache to file
        try:
            with open(core.ERROR_DETECTION_CACHE_FILE, 'w', encoding='utf-8', newline='') as f:
                import csv
                fieldnames = ['cache_key', 'problem_text', 'incorrect_cot', 'error_sentence', 'timestamp', 'api_model']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                cache = core._load_error_detection_cache()
                for cache_key, entry in cache.items():
                    writer.writerow({
                        'cache_key': cache_key,
                        'problem_text': entry['problem_text'],
                        'incorrect_cot': entry['incorrect_cot'],
                        'error_sentence': entry['error_sentence'],
                        'timestamp': entry['timestamp'],
                        'api_model': entry['api_model']
                    })
            core.print_timestamped_message(f"Pre-populated error detection cache with {total_added} entries from existing Excel files")
        except Exception as e:
            core.print_timestamped_message(f"Could not save pre-populated error cache: {e}")
    else:
        core.print_timestamped_message("No new entries to add to error detection cache from Excel files")

def main():
    print("=" * 60)
    print("🔍 ERROR DETECTION CACHE SETUP")
    print("=" * 60)
    print()
    
    print("This script will:")
    print("1. Scan all Excel files in the results/ directory")
    print("2. Extract error analysis results (error sentences)")
    print("3. Pre-populate the error detection cache")
    print("4. Create cache/error_detection_cache.csv")
    print()
    
    # Check if cache already exists
    if os.path.exists(core.ERROR_DETECTION_CACHE_FILE):
        cache = core._load_error_detection_cache()
        print(f"⚠️  Error detection cache file already exists with {len(cache)} entries")
        response = input("Do you want to rebuild it? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted. Error detection cache unchanged.")
            return
        print()
    
    # Run the pre-population
    print("🔄 Pre-populating error detection cache from Excel files...")
    prepopulate_error_cache_from_excel_files()
    
    # Show final status
    if os.path.exists(core.ERROR_DETECTION_CACHE_FILE):
        cache = core._load_error_detection_cache()
        print()
        print("✅ ERROR DETECTION CACHE SETUP COMPLETE!")
        print(f"📁 Cache file: {core.ERROR_DETECTION_CACHE_FILE}")
        print(f"📊 Total entries: {len(cache)}")
        print()
        print("🎯 Benefits:")
        print("  - Error analysis will skip already-analyzed problems")
        print("  - Significant savings on Anthropic API calls")
        print("  - Faster error identification on repeated CoTs")
        print()
        print("🚀 Ready to run experiments with error detection acceleration!")
    else:
        print("❌ Error detection cache setup failed - no cache file created")

if __name__ == "__main__":
    main()
