"""
NNsight-based steering vector implementation for mistake realization.
Uses contrasting pairs of wrong baseline vs successful corrections to build steering vectors.
"""

import torch
import numpy as np
import pandas as pd
import json
import os
import gc
from typing import List, Dict, Tuple, Optional
from cache import print_timestamped_message
from low_level import evaluate_answer


class NNsightSteeringExperiment:
    """
    Steering vector experiment using nnsight for Qwen3 models.
    Builds vectors that encode the direction "realize you've made a mistake".
    """
    
    def __init__(self, model_interface, results_dir: str, layer: int = 16):
        """Initialize steering experiment."""
        self.model_interface = model_interface
        self.results_dir = results_dir
        self.layer = layer
        self.steering_vector = None
        self.train_data = []
        self.validation_data = []
        
    def extract_contrasting_pairs(self, train_fraction: float = 0.8) -> bool:
        """Extract contrasting pairs for steering vector creation."""
        print_timestamped_message("Extracting contrasting pairs for steering vector...")
        
        # Load baseline results (wrong answers)
        baseline_file = os.path.join(self.results_dir, "baseline_results.json")
        if not os.path.exists(baseline_file):
            print_timestamped_message("⚠️ No baseline results found")
            return False
            
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Load intervention results (successful corrections)
        intervention_file = os.path.join(self.results_dir, "intervention_results.json")
        if not os.path.exists(intervention_file):
            print_timestamped_message("⚠️ No intervention results found")
            return False
            
        with open(intervention_file, 'r') as f:
            intervention_data = json.load(f)
        
        # Filter for wrong baseline results that were successfully labeled by Anthropic
        wrong_baseline = [r for r in baseline_data if not r.get('is_correct', True) and r.get('error_location') is not None]
        
        # Filter for successful corrections
        successful_corrections = [r for r in intervention_data if r.get('is_corrected', False)]
        
        if len(wrong_baseline) == 0:
            print_timestamped_message("⚠️ No wrong baseline results with error locations found")
            return False
            
        if len(successful_corrections) == 0:
            print_timestamped_message("⚠️ No successful corrections found")
            return False
            
        # Create contrasting pairs by matching problem IDs
        contrasting_pairs = []
        for correction in successful_corrections:
            problem_id = correction.get('problem_id')
            if problem_id is None:
                continue
                
            # Find corresponding baseline wrong answer
            baseline_match = next((b for b in wrong_baseline if b.get('problem_id') == problem_id), None)
            if baseline_match:
                contrasting_pairs.append({
                    'problem_id': problem_id,
                    'wrong_cot': baseline_match['cot_reasoning'],
                    'correct_cot': correction['cot_reasoning'],
                    'error_location': baseline_match['error_location'],
                    'correct_answer': baseline_match.get('correct_answer', ''),
                    'problem_text': baseline_match.get('problem', '')
                })
        
        if len(contrasting_pairs) < 4:
            print_timestamped_message(f"⚠️ Only {len(contrasting_pairs)} contrasting pairs found, need at least 4")
            return False
            
        print_timestamped_message(f"Found {len(contrasting_pairs)} contrasting pairs")
        
        # Split into training and validation
        np.random.seed(42)
        indices = np.random.permutation(len(contrasting_pairs))
        train_size = int(len(contrasting_pairs) * train_fraction)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        self.train_data = [contrasting_pairs[i] for i in train_indices]
        self.validation_data = [contrasting_pairs[i] for i in val_indices]
        
        print_timestamped_message(f"Split into {len(self.train_data)} training and {len(self.validation_data)} validation pairs")
        return True
    
    def extract_activations(self, prompt: str) -> Optional[np.ndarray]:
        """Extract activations from prompt using nnsight."""
        try:
            nnsight_model = self.model_interface.get_nnsight_model()
            if nnsight_model is None:
                print_timestamped_message("⚠️ Model doesn't support nnsight")
                return None
                
            # Tokenize the prompt
            tokenizer = self.model_interface.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(nnsight_model.device) for k, v in inputs.items()}
            
            # Extract activations using nnsight
            with nnsight_model.trace(inputs, validate=False):
                # Access the transformer layers
                if hasattr(nnsight_model, 'model') and hasattr(nnsight_model.model, 'layers'):
                    layer_output = nnsight_model.model.layers[self.layer]
                elif hasattr(nnsight_model, 'transformer') and hasattr(nnsight_model.transformer, 'h'):
                    layer_output = nnsight_model.transformer.h[self.layer]
                else:
                    # Try direct layer access
                    layer_output = nnsight_model.layers[self.layer]
                
                # Get the output activations (residual stream)
                activations = layer_output.output[0].save()
            
            # Convert to numpy and take last token activations
            activations_np = activations.cpu().numpy()
            if len(activations_np.shape) == 3:
                # Shape: [batch, seq_len, hidden_dim] - take last token
                final_activations = activations_np[0, -1, :]
            else:
                # Shape: [seq_len, hidden_dim] - take last token  
                final_activations = activations_np[-1, :]
                
            return final_activations
            
        except Exception as e:
            print_timestamped_message(f"Error extracting activations: {e}")
            return None
        finally:
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
    
    def build_steering_vector(self) -> bool:
        """Build steering vector from contrasting pairs."""
        if len(self.train_data) == 0:
            return False
            
        print_timestamped_message(f"Building steering vector from {len(self.train_data)} contrasting pairs...")
        
        wrong_activations = []
        correct_activations = []
        
        for i, pair in enumerate(self.train_data):
            print_timestamped_message(f"Processing pair {i+1}/{len(self.train_data)}...")
            
            # Extract activations for wrong reasoning
            wrong_acts = self.extract_activations(pair['wrong_cot'])
            if wrong_acts is not None:
                wrong_activations.append(wrong_acts)
            
            # Extract activations for correct reasoning
            correct_acts = self.extract_activations(pair['correct_cot'])  
            if correct_acts is not None:
                correct_activations.append(correct_acts)
        
        if len(wrong_activations) == 0 or len(correct_activations) == 0:
            print_timestamped_message("⚠️ Failed to extract sufficient activations")
            return False
            
        # Compute steering vector as difference between means
        wrong_mean = np.mean(wrong_activations, axis=0)
        correct_mean = np.mean(correct_activations, axis=0)
        
        self.steering_vector = correct_mean - wrong_mean
        
        # Normalize the steering vector
        vector_norm = np.linalg.norm(self.steering_vector)
        if vector_norm > 0:
            self.steering_vector = self.steering_vector / vector_norm
            
        print_timestamped_message(f"✅ Steering vector built with norm: {vector_norm:.3f}")
        return True
    
    def apply_steering_intervention(self, prompt: str, strength: float = 1.0) -> Optional[str]:
        """Apply steering vector intervention during generation."""
        try:
            nnsight_model = self.model_interface.get_nnsight_model()
            if nnsight_model is None or self.steering_vector is None:
                return None
                
            tokenizer = self.model_interface.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(nnsight_model.device) for k, v in inputs.items()}
            
            # Convert steering vector to tensor
            steering_tensor = torch.tensor(
                self.steering_vector * strength, 
                device=nnsight_model.device, 
                dtype=torch.float32
            )
            
            # Apply steering intervention during generation
            with nnsight_model.trace(inputs, validate=False):
                # Access the layer where we want to apply steering
                if hasattr(nnsight_model, 'model') and hasattr(nnsight_model.model, 'layers'):
                    layer_output = nnsight_model.model.layers[self.layer]
                elif hasattr(nnsight_model, 'transformer') and hasattr(nnsight_model.transformer, 'h'):
                    layer_output = nnsight_model.transformer.h[self.layer]
                else:
                    layer_output = nnsight_model.layers[self.layer]
                
                # Add steering vector to the layer output at the last token position
                layer_activations = layer_output.output[0]
                layer_activations[:, -1, :] += steering_tensor
            
            # Generate continuation with the steered activations
            with torch.no_grad():
                output = nnsight_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode the new tokens
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return response
            
        except Exception as e:
            print_timestamped_message(f"Error applying steering intervention: {e}")
            return None
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    
    def validate_steering(self, strength: float = 1.0) -> Optional[pd.DataFrame]:
        """Apply steering vector to validation data and evaluate performance."""
        if self.steering_vector is None or len(self.validation_data) == 0:
            print_timestamped_message("⚠️ No steering vector or validation data available")
            return None
            
        print_timestamped_message(f"Validating steering on {len(self.validation_data)} examples...")
        
        validation_results = []
        
        for i, pair in enumerate(self.validation_data):
            print_timestamped_message(f"Validating pair {i+1}/{len(self.validation_data)}...")
            
            try:
                # Apply steering intervention to the wrong CoT
                steered_response = self.apply_steering_intervention(pair['wrong_cot'], strength)
                
                if steered_response:
                    # Evaluate the steered response using the existing evaluation system
                    from low_level import smart_evaluate_answer
                    is_correct = smart_evaluate_answer(
                        steered_response, 
                        pair['correct_answer'], 
                        pair['problem_text']
                    )
                    
                    validation_results.append({
                        'problem_id': pair['problem_id'],
                        'original_wrong': pair['wrong_cot'],
                        'steered_response': steered_response,
                        'is_corrected': is_correct,
                        'steering_strength': strength,
                        'layer': self.layer
                    })
                    
            except Exception as e:
                print_timestamped_message(f"Error validating pair {i+1}: {e}")
                
        if validation_results:
            df = pd.DataFrame(validation_results)
            success_rate = (df['is_corrected'].sum() / len(df)) * 100
            print_timestamped_message(f"✅ Steering validation completed: {success_rate:.1f}% success rate")
            return df
        else:
            print_timestamped_message("⚠️ No validation results generated")
            return None


def run_nnsight_steering_experiment(model_interface, results_dir: str, layer: int = 16, steering_strength: float = 1.0) -> Optional[pd.DataFrame]:
    """
    Run the complete nnsight steering experiment.
    
    Args:
        model_interface: Model interface supporting nnsight
        results_dir: Directory containing experiment results
        layer: Layer to extract activations from
        steering_strength: Strength of steering intervention
        
    Returns:
        DataFrame with validation results or None if failed
    """
    print_timestamped_message("🎯 Starting NNsight Steering Vector Experiment")
    print_timestamped_message(f"Target layer: {layer}, Steering strength: {steering_strength}")
    
    # Initialize experiment
    experiment = NNsightSteeringExperiment(model_interface, results_dir, layer)
    
    # Extract contrasting pairs from existing baseline and intervention data
    if not experiment.extract_contrasting_pairs():
        print_timestamped_message("❌ Failed to extract sufficient contrasting pairs")
        return None
    
    # Build steering vector from contrasting activations
    if not experiment.build_steering_vector():
        print_timestamped_message("❌ Failed to build steering vector")
        return None
    
    # Validate steering on held-out data
    results_df = experiment.validate_steering(steering_strength)
    
    if results_df is not None:
        success_rate = (results_df['is_corrected'].sum() / len(results_df)) * 100
        print_timestamped_message(f"🎯 NNsight steering experiment completed: {success_rate:.1f}% success rate")
    else:
        print_timestamped_message("❌ Steering validation failed")
    
    return results_df
