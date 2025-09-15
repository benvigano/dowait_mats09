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
from tqdm import tqdm
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
        
    def extract_contrasting_pairs(self, train_fraction: float = 0.8, max_train_samples: int = None, max_validation_samples: int = None) -> bool:
        """Extract contrasting pairs for steering vector creation."""
        print_timestamped_message("Extracting contrasting pairs for steering vector...")
        
        # Load baseline and intervention results from database
        from database import get_experiment_results
        try:
            all_results = get_experiment_results(self.results_dir)
        except Exception as e:
            print_timestamped_message(f"⚠️ Failed to load results from database: {e}")
            return False
        
        if not all_results:
            print_timestamped_message("⚠️ No results found in database")
            return False
        
        # Separate baseline and intervention results
        baseline_data = []
        intervention_data = []
        
        for result in all_results:
            # Baseline results have baseline_raw_response but no intervention data
            if result.get('baseline_raw_response') and not result.get('intervention_raw_response'):
                baseline_data.append({
                    'problem_id': result.get('rowid', 0),  # Use rowid as problem_id
                    'problem': result.get('problem', ''),
                    'correct_answer': result.get('ground_truth_answer', ''),
                    'cot_reasoning': result.get('baseline_raw_response', ''),
                    'is_correct': bool(result.get('baseline_correct', 0)),
                    'error_line_number': result.get('baseline_error_line_number'),
                    'error_line_content': result.get('baseline_error_line_content'),
                    'error_location': result.get('baseline_error_line_number')  # For compatibility
                })
            # Intervention results have intervention data
            elif result.get('intervention_raw_response'):
                intervention_data.append({
                    'problem_id': result.get('rowid', 0),  # Use rowid as problem_id
                    'problem': result.get('problem', ''),
                    'correct_answer': result.get('ground_truth_answer', ''),
                    'cot_reasoning': result.get('intervention_raw_response', ''),
                    'is_corrected': bool(result.get('intervention_correct', 0))
                })
        
        # Filter for wrong baseline results that have error locations
        wrong_baseline = [r for r in baseline_data if not r.get('is_correct', True) and r.get('error_location') is not None]
        
        # Filter for successful corrections
        successful_corrections = [r for r in intervention_data if r.get('is_corrected', False)]
        
        if len(wrong_baseline) == 0:
            print_timestamped_message("⚠️ No wrong baseline results with error locations found")
            return False
            
        if len(successful_corrections) == 0:
            print_timestamped_message("⚠️ No successful corrections found")
            return False
            
        print_timestamped_message(f"Found {len(wrong_baseline)} wrong baseline results and {len(successful_corrections)} successful corrections")
            
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
        
        print_timestamped_message(f"Found {len(contrasting_pairs)} contrasting pairs")
        
        if len(contrasting_pairs) < 4:
            print_timestamped_message(f"⚠️ Only {len(contrasting_pairs)} contrasting pairs found, need at least 4")
            return False
        
        # Split into training and validation
        np.random.seed(42)
        indices = np.random.permutation(len(contrasting_pairs))
        train_size = int(len(contrasting_pairs) * train_fraction)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Apply sample caps if specified
        if max_train_samples is not None:
            train_indices = train_indices[:max_train_samples]
            print_timestamped_message(f"Limiting training samples to {max_train_samples}")
        
        if max_validation_samples is not None:
            val_indices = val_indices[:max_validation_samples]
            print_timestamped_message(f"Limiting validation samples to {max_validation_samples}")
        
        self.train_data = [contrasting_pairs[i] for i in train_indices]
        self.validation_data = [contrasting_pairs[i] for i in val_indices]
        
        print_timestamped_message(f"Split into {len(self.train_data)} training and {len(self.validation_data)} validation pairs")
        return True
    
    def extract_contrasting_pairs_from_dataframes(self, baseline_df: pd.DataFrame, intervention_df: pd.DataFrame, train_fraction: float = 0.8, max_train_samples: int = None, max_validation_samples: int = None) -> bool:
        """Extract contrasting pairs directly from baseline and intervention DataFrames."""
        print_timestamped_message("Extracting contrasting pairs from provided DataFrames...")
        
        print_timestamped_message(f"Baseline DataFrame: {len(baseline_df)} rows")
        print_timestamped_message(f"Intervention DataFrame: {len(intervention_df)} rows")
        
        # Convert baseline DataFrame to the format needed
        wrong_baseline = []
        for idx, row in baseline_df.iterrows():
            if not row.get('is_correct', True) and row.get('mistake_sentence_usable', False):
                wrong_baseline.append({
                    'problem_id': idx,
                    'problem': row.get('problem', ''),
                    'correct_answer': row.get('ground_truth_answer', ''),
                    'cot_reasoning': row.get('raw_prompt', ''),
                    'is_correct': row.get('is_correct', False),
                    'error_line_number': row.get('error_line_number'),
                    'error_line_content': row.get('error_line_content'),
                    'error_location': row.get('error_line_number')
                })
        
        # Convert intervention DataFrame to the format needed
        successful_corrections = []
        for idx, row in intervention_df.iterrows():
            if row.get('is_corrected', False):
                successful_corrections.append({
                    'problem_id': row.get('problem_id', idx),
                    'problem': row.get('problem', ''),
                    'correct_answer': row.get('correct_answer', ''),
                    'cot_reasoning': row.get('final_prompt', ''),
                    'is_corrected': row.get('is_corrected', False)
                })
        
        print_timestamped_message(f"Found {len(wrong_baseline)} wrong baseline results with error locations")
        print_timestamped_message(f"Found {len(successful_corrections)} successful corrections")
        
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
        
        print_timestamped_message(f"Found {len(contrasting_pairs)} contrasting pairs")
        
        if len(contrasting_pairs) < 4:
            print_timestamped_message(f"⚠️ Only {len(contrasting_pairs)} contrasting pairs found, need at least 4")
            return False
        
        # Split into training and validation
        np.random.seed(42)
        indices = np.random.permutation(len(contrasting_pairs))
        train_size = int(len(contrasting_pairs) * train_fraction)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Apply sample caps if specified
        if max_train_samples is not None:
            train_indices = train_indices[:max_train_samples]
            print_timestamped_message(f"Limiting training samples to {max_train_samples}")
        
        if max_validation_samples is not None:
            val_indices = val_indices[:max_validation_samples]
            print_timestamped_message(f"Limiting validation samples to {max_validation_samples}")
        
        self.train_data = [contrasting_pairs[i] for i in train_indices]
        self.validation_data = [contrasting_pairs[i] for i in val_indices]
        
        print_timestamped_message(f"Split into {len(self.train_data)} training and {len(self.validation_data)} validation pairs")
        return True
    
    def _extract_activations_with_model(self, nnsight_model, prompt: str) -> Optional[np.ndarray]:
        """Extract activations from prompt using a pre-loaded nnsight model."""
        if not prompt or not prompt.strip():
            return None
        try:
            # Tokenize the prompt
            tokenizer = self.model_interface.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            # Ensure indices are integer tensors
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].long()
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].long()
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
            activations_np = activations.detach().cpu().numpy()
            if len(activations_np.shape) == 3:
                # Shape: [batch, seq_len, hidden_dim] - take last token
                return activations_np[0, -1, :]
            elif len(activations_np.shape) == 2:
                # Shape: [seq_len, hidden_dim] - take last token
                return activations_np[-1, :]
            else:
                print_timestamped_message(f"⚠️ Unexpected activation shape: {activations_np.shape}")
                return None
        except Exception as e:
            print_timestamped_message(f"⚠️ Error extracting activations: {e}")
            return None
    
    def extract_activations(self, prompt: str) -> Optional[np.ndarray]:
        """Extract activations from prompt using nnsight."""
        if not prompt or not prompt.strip():
            print(prompt)
            print_timestamped_message("⚠️ Skipping activation extraction for empty prompt.")
            return None
        try:
            nnsight_model = self.model_interface.get_nnsight_model()
            if nnsight_model is None:
                print_timestamped_message("⚠️ Model doesn't support nnsight")
                return None
                
            # Tokenize the prompt
            tokenizer = self.model_interface.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            # Ensure indices are integer tensors
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].long()
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].long()
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
            activations_np = activations.detach().cpu().numpy()
            if len(activations_np.shape) == 3:
                # Shape: [batch, seq_len, hidden_dim] - take last token
                final_activations = activations_np[0, -1, :]
            else:
                # Shape: [seq_len, hidden_dim] - take last token  
                final_activations = activations_np[-1, :]
                
            return final_activations
            
        except Exception as e:
            print_timestamped_message(f"Error extracting activations: {e}")
            import traceback
            print_timestamped_message(f"Detailed error: {traceback.format_exc()}")
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
        
        # Load model once for all extractions
        nnsight_model = self.model_interface.get_nnsight_model()
        if nnsight_model is None:
            print_timestamped_message("⚠️ Model doesn't support nnsight")
            return False
        
        wrong_activations = []
        correct_activations = []
        
        try:
            for pair in tqdm(self.train_data, desc="Building steering vector", unit="pair"):
                # Extract activations for wrong reasoning      
                wrong_acts = self._extract_activations_with_model(nnsight_model, pair['wrong_cot'])
                if wrong_acts is not None:
                    wrong_activations.append(wrong_acts)
                
                # Extract activations for correct reasoning
                correct_acts = self._extract_activations_with_model(nnsight_model, pair['correct_cot'])  
                if correct_acts is not None:
                    correct_activations.append(correct_acts)
        finally:
            # Clean up model after all extractions
            if hasattr(self.model_interface, 'model') and self.model_interface.model is not None:
                from models import drop_model_from_memory
                self.model_interface.model = drop_model_from_memory(self.model_interface.model)
        
        if len(wrong_activations) == 0 or len(correct_activations) == 0:
            print_timestamped_message("⚠️ Failed to extract sufficient activations")
            print_timestamped_message(f"Wrong activations: {len(wrong_activations)}, Correct activations: {len(correct_activations)}")
            return False
            
        # Convert to float64 for more precision and stability
        wrong_activations_64 = [act.astype(np.float64) for act in wrong_activations]
        correct_activations_64 = [act.astype(np.float64) for act in correct_activations]
        
        # Compute steering vector as difference between means
        wrong_mean = np.mean(wrong_activations_64, axis=0)
        correct_mean = np.mean(correct_activations_64, axis=0)
        
        
        # Compute difference with robust handling
        self.steering_vector = correct_mean - wrong_mean
        
        # Check for NaN or inf values and handle them
        if np.any(np.isnan(self.steering_vector)) or np.any(np.isinf(self.steering_vector)):
            print_timestamped_message("⚠️ Steering vector contains NaN or inf values, clipping...")
            self.steering_vector = np.nan_to_num(self.steering_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # If vector is still too large, scale it down
        max_abs_val = np.max(np.abs(self.steering_vector))
        if max_abs_val > 100.0:  # Arbitrary threshold
            print_timestamped_message(f"⚠️ Steering vector has large values (max: {max_abs_val:.3f}), scaling down...")
            self.steering_vector = self.steering_vector / (max_abs_val / 10.0)  # Scale to reasonable range
        
        # Debug: Check vector properties
        vector_norm = np.linalg.norm(self.steering_vector)
        print_timestamped_message(f"Raw steering vector norm: {vector_norm:.3f}")
        print_timestamped_message(f"Steering vector std: {np.std(self.steering_vector):.6f}")
        
        # Normalize the steering vector
        if vector_norm > 0:
            self.steering_vector = self.steering_vector / vector_norm
            print_timestamped_message(f"✅ Steering vector built with normalized norm: {np.linalg.norm(self.steering_vector):.3f}")
        else:
            print_timestamped_message("⚠️ Steering vector has zero norm!")
            return False
            
        return True
    
    def _apply_steering_intervention_with_model(self, nnsight_model, prompt: str, strength: float = 1.0) -> Optional[str]:
        """Apply steering vector intervention during generation using a pre-loaded model."""
        if not prompt or not prompt.strip():
            return None
        try:
            if self.steering_vector is None:
                print_timestamped_message("⚠️ No steering vector available")
                return None

            # Tokenize the prompt
            tokenizer = self.model_interface.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            # Ensure indices are integer tensors
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].long()
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].long()
            inputs = {k: v.to(nnsight_model.device) for k, v in inputs.items()}
            
            # Apply steering intervention during generation
            with nnsight_model.trace(inputs, validate=False) as tracer:
                # Apply steering vector at the target layer
                if hasattr(nnsight_model, 'model') and hasattr(nnsight_model.model, 'layers'):
                    layer_output = nnsight_model.model.layers[self.layer]
                elif hasattr(nnsight_model, 'transformer') and hasattr(nnsight_model.transformer, 'h'):
                    layer_output = nnsight_model.transformer.h[self.layer]
                else:
                    layer_output = nnsight_model.layers[self.layer]
                
                layer_activations = layer_output.output[0]
                
                # Convert steering vector to tensor and apply
                steering_tensor = torch.tensor(self.steering_vector, dtype=layer_activations.dtype, device=layer_activations.device)
                steering_tensor = steering_tensor * strength
                
                # Handle different tensor shapes correctly
                if len(layer_activations.shape) == 3:
                    # Shape: [batch, seq_len, hidden_dim] - apply to last token
                    layer_activations[:, -1, :] += steering_tensor
                elif len(layer_activations.shape) == 2:
                    # Shape: [seq_len, hidden_dim] - apply to last token
                    layer_activations[-1, :] += steering_tensor
                else:
                    print_timestamped_message(f"⚠️ Unexpected layer activation shape: {layer_activations.shape}")
                    return None
            
            # Generate with steering applied
            max_new_tokens = 1000
            
            outputs = nnsight_model.generate(
                inputs['input_ids'], 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.get('attention_mask')
            )
            
            # Decode the response (excluding the original prompt)
            if outputs is not None and len(outputs) > 0:
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]  # Remove original prompt
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                return response
            
            return None
        except Exception as e:
            print_timestamped_message(f"⚠️ Error in steering intervention: {e}")
            return None
    
    def apply_steering_intervention(self, prompt: str, strength: float = 1.0) -> Optional[str]:
        """Apply steering vector intervention during generation."""
        if not prompt or not prompt.strip():
            print_timestamped_message("⚠️ Skipping steering intervention for empty prompt.")
            return None
        try:
            nnsight_model = self.model_interface.get_nnsight_model()
            if nnsight_model is None or self.steering_vector is None:
                print_timestamped_message("⚠️ No nnsight model or steering vector available")
                return None
                
            tokenizer = self.model_interface.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            # Ensure indices are integer tensors
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].long()
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].long()
            inputs = {k: v.to(nnsight_model.device) for k, v in inputs.items()}
            
            
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
                steering_tensor = torch.tensor(
                    self.steering_vector * strength,
                    device=layer_activations.device,
                    dtype=layer_activations.dtype
                )
                
                
                # Handle different tensor shapes correctly
                if len(layer_activations.shape) == 3:
                    # Shape: [batch, seq_len, hidden_dim] - apply to last token
                    layer_activations[:, -1, :] += steering_tensor
                elif len(layer_activations.shape) == 2:
                    # Shape: [seq_len, hidden_dim] - apply to last token
                    layer_activations[-1, :] += steering_tensor
                else:
                    print_timestamped_message(f"⚠️ Unexpected layer activation shape: {layer_activations.shape}")
                    return None
                
            
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
            import traceback
            print_timestamped_message(f"Detailed error: {traceback.format_exc()}")
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
        
        # Load model once for all validations
        nnsight_model = self.model_interface.get_nnsight_model()
        if nnsight_model is None:
            print_timestamped_message("⚠️ Model doesn't support nnsight")
            return None
        
        validation_results = []
        
        try:
            for pair in tqdm(self.validation_data, desc="Validating steering", unit="pair"):
                try:
                    # Apply steering intervention to the wrong CoT
                    steered_response = self._apply_steering_intervention_with_model(nnsight_model, pair['wrong_cot'], strength)
                    
                    if steered_response:
                        # Evaluate the steered response using the existing evaluation system
                        from low_level import evaluate_answer
                        evaluation_result = evaluate_answer(
                            pair['problem_text'],
                            pair['correct_answer'], 
                            steered_response
                        )
                        is_correct = evaluation_result['is_correct']
                        
                        validation_results.append({
                            'problem_id': pair['problem_id'],
                            'original_wrong': pair['wrong_cot'],
                            'steered_response': steered_response,
                            'is_corrected': is_correct,
                            'steering_strength': strength,
                            'layer': self.layer
                        })
                    
                except Exception as e:
                    import traceback
                    print_timestamped_message(f"Detailed error: {traceback.format_exc()}")
                    continue
        finally:
            # Clean up model after all validations
            if hasattr(self.model_interface, 'model') and self.model_interface.model is not None:
                from models import drop_model_from_memory
                self.model_interface.model = drop_model_from_memory(self.model_interface.model)
                
        if validation_results:
            df = pd.DataFrame(validation_results)
            success_rate = (df['is_corrected'].sum() / len(df)) * 100
            print_timestamped_message(f"✅ Steering validation completed: {success_rate:.1f}% success rate")
            return df
        else:
            print_timestamped_message("⚠️ No validation results generated")
            return None


def run_nnsight_steering_experiment(model_interface, results_dir: str, layer: int = 16, steering_strength: float = 1.0, baseline_data: pd.DataFrame = None, intervention_data: pd.DataFrame = None, max_train_samples: int = None, max_validation_samples: int = None) -> Optional[pd.DataFrame]:
    """
    Run the complete nnsight steering experiment.
    
    Args:
        model_interface: Model interface supporting nnsight
        results_dir: Directory containing experiment results
        layer: Layer to extract activations from
        steering_strength: Strength of steering intervention
        baseline_data: DataFrame with baseline experiment results (direct from experiment)
        intervention_data: DataFrame with intervention experiment results (direct from experiment)
        max_train_samples: Maximum number of training samples to use (None for unlimited)
        max_validation_samples: Maximum number of validation samples to use (None for unlimited)
        
    Returns:
        DataFrame with validation results or None if failed
    """
    print_timestamped_message("🎯 Starting NNsight Steering Vector Experiment")
    print_timestamped_message(f"Using model: {model_interface.model_id}")
    print_timestamped_message(f"Target layer: {layer}, Steering strength: {steering_strength}")
    
    # Initialize experiment
    experiment = NNsightSteeringExperiment(model_interface, results_dir, layer)

    
    # Extract contrasting pairs from provided data or database
    if baseline_data is not None and intervention_data is not None:
        print_timestamped_message("Using provided baseline and intervention data")
        if not experiment.extract_contrasting_pairs_from_dataframes(baseline_data, intervention_data, max_train_samples=max_train_samples, max_validation_samples=max_validation_samples):
            print_timestamped_message("❌ Failed to extract sufficient contrasting pairs from provided data")
            return None
    else:
        print_timestamped_message("Loading baseline and intervention data from database")
        if not experiment.extract_contrasting_pairs(max_train_samples=max_train_samples, max_validation_samples=max_validation_samples):
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
