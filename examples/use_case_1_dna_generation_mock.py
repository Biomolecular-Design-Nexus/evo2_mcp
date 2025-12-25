#!/usr/bin/env python3
"""
Use Case 1: DNA Sequence Generation with Evo2 (Mock Version)

Description: Mock version for CPU-only environment that demonstrates the CLI and data processing
without requiring the actual Evo2 models which need CUDA.

Input: DNA sequence prompts
Output: Mock generated DNA sequences
Complexity: Simple
Source: notebooks/generation/generation_notebook.ipynb
Priority: High
Environment: ./env (CPU-only compatible)
"""

import argparse
import random
import time
from typing import List

def mock_generate_dna_sequences(
    prompts: List[str],
    model_name: str = 'evo2_7b',
    n_tokens: int = 500,
    temperature: float = 1.0,
    top_k: int = 4
) -> List[str]:
    """
    Mock DNA sequence generation that simulates the Evo2 model behavior.

    Args:
        prompts: List of DNA sequence prompts
        model_name: Evo2 model to use (for display only)
        n_tokens: Number of tokens to generate
        temperature: Sampling temperature (affects randomness simulation)
        top_k: Top-k sampling parameter (affects randomness simulation)

    Returns:
        List of mock generated DNA sequences
    """
    print(f"[MOCK] Loading Evo2 model: {model_name}")
    print(f"[MOCK] Model would require GPU/CUDA for actual inference")

    # Simulate model loading time
    time.sleep(1)

    print(f"[MOCK] Generating sequences for {len(prompts)} prompts...")

    # DNA nucleotides
    nucleotides = ['A', 'T', 'C', 'G']
    sequences = []

    for i, prompt in enumerate(prompts):
        # Set seed based on prompt for reproducible mock results
        random.seed(hash(prompt) % 1000000)

        # Start with the prompt
        sequence = prompt.upper()

        # Generate additional nucleotides
        remaining_tokens = n_tokens - len(prompt)
        if remaining_tokens > 0:
            # Simulate temperature effect on randomness
            if temperature < 0.5:
                # Lower temperature = more deterministic (mostly A and T)
                weights = [0.4, 0.4, 0.1, 0.1]
            elif temperature > 1.5:
                # Higher temperature = more random
                weights = [0.25, 0.25, 0.25, 0.25]
            else:
                # Moderate temperature
                weights = [0.3, 0.3, 0.2, 0.2]

            for _ in range(remaining_tokens):
                sequence += random.choices(nucleotides, weights=weights)[0]

        sequences.append(sequence)
        print(f"[MOCK] Generated sequence {i+1}: {len(sequence)} bp")

    return sequences

def mock_generate_species_specific_sequence(
    species: str,
    n_tokens: int = 500,
    model_name: str = 'evo2_7b'
) -> str:
    """
    Mock species-specific sequence generation.

    Args:
        species: Species name (e.g., 'Homo sapiens')
        n_tokens: Number of tokens to generate
        model_name: Evo2 model to use (for display only)

    Returns:
        Mock generated DNA sequence
    """
    print(f"[MOCK] Creating phylogenetic tag for species: {species}")

    # Mock species tag (actual implementation uses GBIF taxonomy)
    species_tag = f"<{species.replace(' ', '_').lower()}>"
    print(f"[MOCK] Species tag: {species_tag}")

    # Generate species-specific mock sequence
    random.seed(hash(species) % 1000000)

    # Species-specific nucleotide preferences (completely mock)
    if 'homo' in species.lower() or 'human' in species.lower():
        # Mock human-like GC content (~40%)
        weights = [0.3, 0.3, 0.2, 0.2]  # A, T, C, G
    elif 'escherichia' in species.lower() or 'coli' in species.lower():
        # Mock E. coli-like GC content (~50%)
        weights = [0.25, 0.25, 0.25, 0.25]
    else:
        # Default weights
        weights = [0.25, 0.25, 0.25, 0.25]

    nucleotides = ['A', 'T', 'C', 'G']
    sequence = ""
    for _ in range(n_tokens):
        sequence += random.choices(nucleotides, weights=weights)[0]

    return sequence

def main():
    parser = argparse.ArgumentParser(description="Generate DNA sequences with Evo2 (Mock Version)")
    parser.add_argument('--prompts', nargs='+',
                       default=['ACGT', 'ATCGATCG'],
                       help='DNA sequence prompts')
    parser.add_argument('--model', default='evo2_7b',
                       help='Evo2 model to use')
    parser.add_argument('--tokens', type=int, default=100,
                       help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--species', type=str,
                       help='Generate sequence for specific species')
    parser.add_argument('--output', type=str,
                       help='Output file to save sequences')

    args = parser.parse_args()

    print("=" * 60)
    print("MOCK VERSION - Evo2 DNA Generation Demo")
    print("This version runs without GPU/CUDA requirements")
    print("Actual Evo2 models require transformer_engine and CUDA")
    print("=" * 60)

    # Set random seed for reproducibility
    random.seed(42)

    if args.species:
        print(f"[MOCK] Generating sequence for species: {args.species}")
        sequence = mock_generate_species_specific_sequence(
            args.species, args.tokens, args.model
        )
        sequences = [sequence]
    else:
        print(f"[MOCK] Generating sequences for prompts: {args.prompts}")
        sequences = mock_generate_dna_sequences(
            args.prompts, args.model, args.tokens, args.temperature
        )

    # Display results
    print("\n" + "=" * 50)
    print("GENERATED SEQUENCES:")
    print("=" * 50)
    for i, seq in enumerate(sequences):
        print(f"\nSequence {i+1}:")
        print(f"Length: {len(seq)} bp")
        print(f"GC Content: {((seq.count('G') + seq.count('C')) / len(seq) * 100):.1f}%")
        print(f"Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
        if len(seq) > 50:
            print(f"          {'...' + seq[-47:]}")

    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">Generated_sequence_{i+1}\n{seq}\n")
        print(f"\n[MOCK] Sequences saved to {args.output}")

    print("\n" + "=" * 50)
    print("MOCK EXECUTION COMPLETED")
    print("For actual Evo2 inference, set up GPU environment with CUDA")
    print("=" * 50)

if __name__ == "__main__":
    main()