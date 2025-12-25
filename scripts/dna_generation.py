#!/usr/bin/env python3
"""
Script: dna_generation.py
Description: DNA sequence generation and autocompletion using Evo2 models

Original Use Case: examples/use_case_1_dna_generation.py (and mock version)
Dependencies Removed: torch imports inlined, evo2 imports with lazy loading

Usage:
    python scripts/dna_generation.py --input <prompts> --output <output_file>

Example:
    python scripts/dna_generation.py --prompts ACGT ATCG --tokens 100 --output results/sequences.fasta
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import random
import time
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import sys

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model_name": "evo2_7b",
    "n_tokens": 100,
    "temperature": 1.0,
    "top_k": 4,
    "use_gpu": False,  # Set to True when GPU available
    "mock_mode": True  # Set to False for actual Evo2 inference
}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of DNA sequence."""
    if not sequence:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100

def format_sequence_display(sequence: str, max_display: int = 50) -> str:
    """Format sequence for display with truncation."""
    if len(sequence) <= max_display:
        return sequence
    return f"{sequence[:max_display]}...{sequence[-max_display:]}"

def save_fasta(sequences: List[str], file_path: Path, headers: Optional[List[str]] = None) -> None:
    """Save sequences to FASTA file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        for i, seq in enumerate(sequences):
            if headers and i < len(headers):
                header = headers[i]
            else:
                header = f"Generated_sequence_{i+1}"
            f.write(f">{header}\n{seq}\n")

def create_phylogenetic_tag(species: str) -> str:
    """Create phylogenetic tag for species-specific generation."""
    return f"<{species.replace(' ', '_').lower()}>"

# ==============================================================================
# Mock Generation Functions (for CPU-only environment)
# ==============================================================================
def mock_generate_dna_sequences(
    prompts: List[str],
    config: Dict[str, Any]
) -> List[str]:
    """
    Mock DNA sequence generation that simulates Evo2 model behavior.
    """
    print(f"[MOCK] Loading Evo2 model: {config['model_name']}")
    print(f"[MOCK] Model would require GPU/CUDA for actual inference")

    time.sleep(1)  # Simulate loading time
    print(f"[MOCK] Generating sequences for {len(prompts)} prompts...")

    nucleotides = ['A', 'T', 'C', 'G']
    sequences = []

    for i, prompt in enumerate(prompts):
        # Set seed based on prompt for reproducible results
        random.seed(hash(prompt) % 1000000)

        sequence = prompt.upper()
        remaining_tokens = config['n_tokens'] - len(prompt)

        if remaining_tokens > 0:
            # Temperature effect on randomness
            temp = config['temperature']
            if temp < 0.5:
                weights = [0.4, 0.4, 0.1, 0.1]  # Low temp: more deterministic
            elif temp > 1.5:
                weights = [0.25, 0.25, 0.25, 0.25]  # High temp: more random
            else:
                weights = [0.3, 0.3, 0.2, 0.2]  # Moderate temp

            for _ in range(remaining_tokens):
                sequence += random.choices(nucleotides, weights=weights)[0]

        sequences.append(sequence)
        print(f"[MOCK] Generated sequence {i+1}: {len(sequence)} bp")

    return sequences

def mock_generate_species_specific(species: str, config: Dict[str, Any]) -> str:
    """Mock species-specific sequence generation."""
    print(f"[MOCK] Creating phylogenetic tag for species: {species}")

    species_tag = create_phylogenetic_tag(species)
    print(f"[MOCK] Species tag: {species_tag}")

    random.seed(hash(species) % 1000000)

    # Species-specific nucleotide preferences (mock implementation)
    if 'homo' in species.lower() or 'human' in species.lower():
        weights = [0.3, 0.3, 0.2, 0.2]  # Mock human-like GC content
    elif 'escherichia' in species.lower() or 'coli' in species.lower():
        weights = [0.25, 0.25, 0.25, 0.25]  # Mock E. coli-like GC content
    else:
        weights = [0.25, 0.25, 0.25, 0.25]  # Default

    nucleotides = ['A', 'T', 'C', 'G']
    sequence = ""
    for _ in range(config['n_tokens']):
        sequence += random.choices(nucleotides, weights=weights)[0]

    return sequence

# ==============================================================================
# Real Evo2 Functions (lazy loaded when GPU available)
# ==============================================================================
def get_evo2_model(model_name: str):
    """Lazy load Evo2 model to minimize startup time."""
    try:
        # Add repo to path only when needed
        repo_path = Path(__file__).parent.parent / "repo"
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from evo2 import Evo2
        from evo2.utils import make_phylotag_from_gbif

        return Evo2(model_name), make_phylotag_from_gbif
    except Exception as e:
        print(f"Warning: Could not load Evo2 model: {e}")
        print("Falling back to mock mode")
        return None, None

def real_generate_dna_sequences(
    prompts: List[str],
    config: Dict[str, Any]
) -> List[str]:
    """Real DNA sequence generation using Evo2."""
    model, _ = get_evo2_model(config['model_name'])
    if model is None:
        print("Falling back to mock generation")
        return mock_generate_dna_sequences(prompts, config)

    print(f"Loading Evo2 model: {config['model_name']}")
    sequences = []

    for prompt in prompts:
        sequence = model.generate(
            prompt,
            n_tokens=config['n_tokens'],
            temperature=config['temperature'],
            top_k=config['top_k']
        )
        sequences.append(sequence)

    return sequences

def real_generate_species_specific(species: str, config: Dict[str, Any]) -> str:
    """Real species-specific sequence generation."""
    model, make_phylotag = get_evo2_model(config['model_name'])
    if model is None:
        print("Falling back to mock generation")
        return mock_generate_species_specific(species, config)

    # Create phylogenetic tag
    species_tag = make_phylotag(species)

    # Generate sequence with species context
    sequence = model.generate(
        species_tag,
        n_tokens=config['n_tokens'],
        temperature=config['temperature'],
        top_k=config['top_k']
    )

    return sequence

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_dna_generation(
    prompts: Optional[List[str]] = None,
    species: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for DNA sequence generation.

    Args:
        prompts: List of DNA sequence prompts (if None and species is None, uses default)
        species: Species name for species-specific generation
        output_file: Path to save output FASTA file (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - sequences: Generated DNA sequences
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_dna_generation(prompts=["ACGT", "ATCG"], output_file="output.fasta")
        >>> print(result['sequences'])
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if prompts is None and species is None:
        prompts = ['ACGT', 'ATCGATCG']  # Default prompts

    # Set random seed for reproducibility in mock mode
    if config['mock_mode']:
        random.seed(42)

    # Generate sequences
    if species:
        print(f"Generating sequence for species: {species}")
        if config['mock_mode']:
            sequence = mock_generate_species_specific(species, config)
        else:
            sequence = real_generate_species_specific(species, config)
        sequences = [sequence]
        headers = [f"{species}_generated"]
    else:
        print(f"Generating sequences for prompts: {prompts}")
        if config['mock_mode']:
            sequences = mock_generate_dna_sequences(prompts, config)
        else:
            sequences = real_generate_dna_sequences(prompts, config)
        headers = [f"prompt_{i+1}_{prompts[i][:10]}" for i in range(len(prompts))]

    # Calculate statistics
    stats = []
    for seq in sequences:
        stats.append({
            'length': len(seq),
            'gc_content': calculate_gc_content(seq)
        })

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        save_fasta(sequences, output_path, headers)

    return {
        "sequences": sequences,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "config": config,
            "species": species,
            "prompts": prompts,
            "statistics": stats
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--prompts', nargs='+',
                       help='DNA sequence prompts (default: ACGT ATCGATCG)')
    parser.add_argument('--species', type=str,
                       help='Generate sequence for specific species')
    parser.add_argument('--model', default='evo2_7b',
                       help='Evo2 model to use (default: evo2_7b)')
    parser.add_argument('--tokens', type=int, default=100,
                       help='Number of tokens to generate (default: 100)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top-k', type=int, default=4,
                       help='Top-k sampling parameter (default: 4)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output FASTA file path')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for generated files')
    parser.add_argument('--config', '-c', type=str,
                       help='Config file (JSON)')
    parser.add_argument('--real-mode', action='store_true',
                       help='Use real Evo2 models (requires GPU)')
    parser.add_argument('--batch-mode', action='store_true',
                       help='Run in batch mode for multiple files')
    parser.add_argument('--prompt-files', nargs='+',
                       help='Files containing prompts (one per line)')
    parser.add_argument('--max-sequences', type=int,
                       help='Maximum number of sequences to process')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Handle output file/directory
    output_file = args.output
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if not output_file:
            output_file = str(Path(args.output_dir) / "dna_sequences.fasta")

    # Override config with command line args
    overrides = {
        'model_name': args.model,
        'n_tokens': args.tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'mock_mode': not args.real_mode
    }

    # Print header
    print("=" * 60)
    if overrides['mock_mode']:
        print("MOCK VERSION - Evo2 DNA Generation Demo")
        print("This version runs without GPU/CUDA requirements")
        print("Use --real-mode for actual Evo2 models (requires GPU)")
    else:
        print("Evo2 DNA Generation")
        print("Using actual Evo2 models (requires GPU/CUDA)")
    print("=" * 60)

    # Run generation
    result = run_dna_generation(
        prompts=args.prompts,
        species=args.species,
        output_file=output_file,
        config=config,
        **overrides
    )

    # Display results
    print("\n" + "=" * 50)
    print("GENERATED SEQUENCES:")
    print("=" * 50)

    for i, seq in enumerate(result['sequences']):
        stats = result['metadata']['statistics'][i]
        print(f"\nSequence {i+1}:")
        print(f"Length: {stats['length']} bp")
        print(f"GC Content: {stats['gc_content']:.1f}%")
        print(f"Sequence: {format_sequence_display(seq)}")

    if result['output_file']:
        print(f"\nSequences saved to: {result['output_file']}")

    print("\n" + "=" * 50)
    if overrides['mock_mode']:
        print("MOCK EXECUTION COMPLETED")
        print("For actual Evo2 inference, use --real-mode with GPU environment")
    else:
        print("EXECUTION COMPLETED")
    print("=" * 50)

    # Save JSON output for MCP job manager if specified by args
    if hasattr(args, 'output') and str(output_file).endswith('.json'):
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

    return result

if __name__ == '__main__':
    main()