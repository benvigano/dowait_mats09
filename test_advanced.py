#!/usr/bin/env python3
"""
Simple test script to demonstrate advanced GPU testing features.
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Run various GPU tests."""
    print("🎯 RTX A6000 Advanced GPU Test Suite")
    print("=" * 60)
    
    # Test 1: Basic GPU test
    print("\n1️⃣ Running basic GPU test...")
    success1 = run_command("source $HOME/.local/bin/env && uv run python gpu_test.py")
    
    # Test 2: Advanced test with different model
    print("\n2️⃣ Running advanced test with GPT-2 Large...")
    success2 = run_command("source $HOME/.local/bin/env && uv run python advanced_gpu_test.py --model gpt2-large --prompt 'The future of AI is'")
    
    # Test 3: Multiple models test (commented out to avoid long runtime)
    print("\n3️⃣ Multiple models test (skipped - uncomment to run)")
    # success3 = run_command("source $HOME/.local/bin/env && uv run python advanced_gpu_test.py --multi")
    
    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print("=" * 60)
    
    if success1 and success2:
        print("🎉 All tests passed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()
