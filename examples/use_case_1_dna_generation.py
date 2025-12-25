#!/usr/bin/env python3
"""
Use Case 1: DNA Sequence Generation with Evo2

Description: Generate DNA sequences based on prompts using Evo2 models.
This script demonstrates how to perform DNA autocompletion and generation.

Input: DNA sequence prompts
Output: Generated DNA sequences
Complexity: Simple
Source: notebooks/generation/generation_notebook.ipynb
Priority: High
Environment: ./env
"""

import argparse
import torch
from typing import List, Optional

# Note: Import will fail if environment is not properly set up
try:
    from evo2 import Evo2
    from evo2.utils import make_phylotag_from_gbif
except ImportError as e:
    print(f"Error importing evo2: {e}")
    print("Please ensure the environment is properly set up with Evo2 dependencies")
    exit(1)


def generate_dna_sequences(
    prompts: List[str],
    model_name: str = 'evo2_7b',
    n_tokens: int = 500,
    temperature: float = 1.0,
    top_k: int = 4
) -> List[str]:
    """
    Generate DNA sequences using Evo2 model.

    Args:
        prompts: List of DNA sequence prompts
        model_name: Evo2 model to use
        n_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter

    Returns:
        List of generated DNA sequences
    """
    print(f"Loading Evo2 model: {model_name}")
    model = Evo2(model_name)

    print(f"Generating sequences for {len(prompts)} prompts...")
    generations = model.generate(
        prompts,
        n_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k
    )

    return generations.sequences


def generate_species_specific_sequence(
    species: str,
    n_tokens: int = 500,
    model_name: str = 'evo2_7b'
) -> str:
    """
    Generate a sequence for a specific species using phylogenetic tagging.

    Args:
        species: Species name (e.g., 'Homo sapiens')
        n_tokens: Number of tokens to generate
        model_name: Evo2 model to use

    Returns:
        Generated DNA sequence
    """
    # Create species-specific prompt
    species_tag = make_phylotag_from_gbif(species)
    print(f"Species tag for {species}: {species_tag}")

    model = Evo2(model_name)
    generation = model.generate(
        [species_tag],
        n_tokens=n_tokens,
        temperature=1.0
    )

    return generation.sequences[0]


def main():
    parser = argparse.ArgumentParser(description="Generate DNA sequences with Evo2")
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

    # Set random seed for reproducibility
    torch.manual_seed(42)

    if args.species:
        print(f"Generating sequence for species: {args.species}")
        sequence = generate_species_specific_sequence(
            args.species, args.tokens, args.model
        )
        sequences = [sequence]
    else:
        print(f"Generating sequences for prompts: {args.prompts}")
        sequences = generate_dna_sequences(
            args.prompts, args.model, args.tokens, args.temperature
        )

    # Display results
    for i, seq in enumerate(sequences):
        print(f"\nGenerated sequence {i+1}:")
        print(seq)

    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">Generated_sequence_{i+1}\n{seq}\n")
        print(f"\nSequences saved to {args.output}")


if __name__ == "__main__":
    main()