"""
AutoSteer - The Feature Hunter Agent for LLM Interpretability
Author: Dr. Ellie Vance Persona
Description: Automates the extraction of steering vectors via Activation Addition.
"""

import torch
# Monkeypatch for transformer_lens compatibility with newer transformers
import transformers
import os
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from transformer_lens import HookedTransformer
from typing import List, Tuple
import torch.nn.functional as F
import einops

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
TARGET_LAYER = 6
STEERING_COEFF = 5.0 # "Strength" of the injection

class AutoSteer:
    def __init__(self):
        print(f"--> [Init] Loading {MODEL_NAME} on {DEVICE}...")
        # HookedTransformer wraps the HF model and exposes 
        # internal states via string-based hooks (e.g., 'blocks.6.hook_resid_pre').
        self.model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
        self.steering_vector = None
        self.hook_point = f"blocks.{TARGET_LAYER}.hook_resid_pre"

    def _get_mean_activations(self, texts: List[str]) -> torch.Tensor:
        """
        Runs the model and extracts the mean activation at the target layer.
        """
        # Run with cache enabled to capture the residual stream
        # Shape: [batch, seq_len, d_model]
        _, cache = self.model.run_with_cache(texts, names_filter=[self.hook_point])
        acts = cache[self.hook_point]

        # Average across the sequence length (dim=1) to get a 
        # "summary vector" for the sentence. Ideally, we would grab the last token
        # for prompt steering, but mean-pooling is more robust for general "tone".
        # acts shape: [batch, seq_len, d_model]
        return einops.reduce(acts, "batch seq d_model -> d_model", "mean")

    def extract_vector(self, positive_examples: List[str], negative_examples: List[str]):
        """
        Calculates the steering vector: Center(Positive) - Center(Negative).
        """
        print(f"--> [Analysis] Extracting vectors from {len(positive_examples)} pairs...")
        
        pos_mean = self._get_mean_activations(positive_examples)
        neg_mean = self._get_mean_activations(negative_examples)

        # The core operation: Difference of Means
        direction = pos_mean - neg_mean
        
        # normalize to unit length so that 'STEERING_COEFF' 
        # has a consistent meaning regardless of the vector's raw magnitude.
        self.steering_vector = direction / direction.norm()
        print("--> [Success] Steering vector isolated and normalized.")

    def _hook_function(self, resid_pre, hook):
        """
        The Intervention: Modifies the residual stream in-flight.
        """
        if self.steering_vector is not None:
            # Broadcast addition: [batch, seq_len, d_model] + [d_model]
            resid_pre += self.steering_vector * STEERING_COEFF
        return resid_pre

    def generate(self, prompt: str, steer: bool = False):
        print(f"\n--> [Gen] Prompt: '{prompt}' | Steered: {steer}")
        
        hooks = []
        if steer:
            hooks = [(self.hook_point, self._hook_function)]

        # Context manager applies hooks ONLY for this generation block
        with self.model.hooks(fwd_hooks=hooks):
            output = self.model.generate(
                prompt, 
                max_new_tokens=25, 
                temperature=0.7, 
                verbose=False
            )
        print(f"    Output: {output}")

if __name__ == "__main__":
    # 1. Setup
    agent = AutoSteer()

    # 2. Data (The "Experiment")
    pirate_texts = [
        "Arr matey, give me the gold!",
        "Shiver me timbers, the sea is rough.",
        "Aye captain, we sail at dawn."
    ]
    neutral_texts = [
        "Hello friend, give me the money.",
        "Wow, the ocean is very choppy.",
        "Yes boss, we leave in the morning."
    ]

    # 3. Extraction
    agent.extract_vector(pirate_texts, neutral_texts)

    # 4. Test (The Proof)
    test_prompt = "I went to the grocery store and"
    agent.generate(test_prompt, steer=False) # Baseline
    agent.generate(test_prompt, steer=True)  # Intervention
