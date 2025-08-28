# RTX A6000 GPU Memory Test

This project contains a comprehensive GPU memory test script designed to test your RTX A6000 with 62GB memory. The script loads a large language model and performs inference to demonstrate GPU memory utilization.

## Features

- **GPU Information Display**: Shows detailed information about your GPU(s)
- **Memory Monitoring**: Tracks both GPU and system memory usage throughout the process
- **Large Model Loading**: Loads Microsoft's DialoGPT-large model (~1.5GB) to test memory capacity
- **Inference Testing**: Performs text generation with "Hello, world!" prompt
- **Memory Cleanup**: Properly cleans up memory after testing

## Requirements

- RTX A6000 or compatible GPU
- CUDA support
- Python 3.11+
- 62GB+ system memory (as specified in your runpod.io setup)

## Installation

The project uses `uv` for package management. All dependencies are already configured in `pyproject.toml`.

## Usage

Run the GPU test script:

```bash
source $HOME/.local/bin/env && uv run python gpu_test.py
```

## What the Script Does

1. **GPU Detection**: Checks for CUDA availability and displays GPU information
2. **Memory Baseline**: Shows initial memory usage
3. **Model Loading**: Downloads and loads the DialoGPT-large model (~1.5GB)
4. **Memory Check**: Shows memory usage after model loading
5. **Inference**: Runs text generation with "Hello, world!" prompt
6. **Cleanup**: Properly releases memory and shows final status

## Expected Output

The script will display:
- GPU information (name, memory capacity)
- Memory usage at each stage
- Model loading progress
- Generated text response
- Performance metrics (loading time, generation time)

## Model Details

- **Model**: `microsoft/DialoGPT-large`
- **Size**: ~1.5GB
- **Type**: Causal Language Model
- **Precision**: Float16 (for memory efficiency)

## Troubleshooting

If you encounter issues:

1. **CUDA not available**: Ensure NVIDIA drivers are properly installed
2. **Out of memory**: The script uses memory-efficient loading, but you can try smaller models
3. **Download issues**: Check your internet connection for model download

## Customization

You can modify the script to:
- Test different models by changing the `model_name` parameter
- Adjust generation parameters (temperature, max_length, etc.)
- Add more memory-intensive operations
- Test with multiple models simultaneously

## Performance Notes

- The RTX A6000 has 48GB VRAM, so this test should run smoothly
- The script uses half-precision (float16) to optimize memory usage
- Automatic device mapping ensures optimal GPU utilization
