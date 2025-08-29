#!/usr/bin/env python3
"""
One-time cache setup script to pre-populate the generation cache from existing Excel files.
Run this once to extract all previous baseline generations into the cache.

Usage: uv run setup_cache.py
"""

import core
import os

def main():
    print("=" * 60)
    print("🚀 GENERATION CACHE SETUP")
    print("=" * 60)
    print()
    
    print("This script will:")
    print("1. Scan all Excel files in the results/ directory")
    print("2. Extract baseline generation results")
    print("3. Extract intervention generation results")
    print("4. Pre-populate the generation cache")
    print("5. Create cache/generation_cache.csv")
    print()
    
    # Check if cache already exists
    if os.path.exists(core.GENERATION_CACHE_FILE):
        cache = core._load_generation_cache()
        print(f"⚠️  Cache file already exists with {len(cache)} entries")
        response = input("Do you want to rebuild it? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted. Cache unchanged.")
            return
        print()
    
    # Run the pre-population
    print("🔄 Pre-populating cache from Excel files...")
    core.prepopulate_cache_from_excel_files_enhanced()
    
    # Show final status
    if os.path.exists(core.GENERATION_CACHE_FILE):
        cache = core._load_generation_cache()
        print()
        print("✅ CACHE SETUP COMPLETE!")
        print(f"📁 Cache file: {core.GENERATION_CACHE_FILE}")
        print(f"📊 Total entries: {len(cache)}")
        print()
        print("🎯 Benefits:")
        print("  - Baseline experiments will skip already-generated problems")
        print("  - Insertion tests will reuse cached baseline generations")
        print("  - Significant speedup on repeated problem sets")
        print()
        print("🚀 Ready to run experiments with cache acceleration!")
    else:
        print("❌ Cache setup failed - no cache file created")

if __name__ == "__main__":
    main()
