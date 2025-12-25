#!/usr/bin/env python3
"""
Script: phage_genome_design.py
Description: Novel bacteriophage genome design and filtering using Evo2 models

Original Use Case: examples/use_case_4_phage_genome_design.py
Dependencies Removed: subprocess calls simplified, evo2 imports with lazy loading

Usage:
    python scripts/phage_genome_design.py --reference <file> --output <output_file>

Example:
    python scripts/phage_genome_design.py --reference examples/data/NC_001422_1.fna --output results/designed_genomes.fasta
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import tempfile
import random
import time
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
import json
import sys

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model_name": "evo2_7b",
    "num_designs": 5,
    "design_length": 5000,
    "mutation_rate": 0.1,
    "filter_criteria": {
        "min_length": 1000,
        "max_length": 10000,
        "min_gc_content": 20,
        "max_gc_content": 80,
        "max_n_content": 5
    },
    "generation_params": {
        "temperature": 1.2,
        "top_k": 8,
        "window_size": 1024
    },
    "mock_mode": True
}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def load_reference_genome(file_path: Union[str, Path]) -> Tuple[str, str]:
    """Load reference genome from FASTA file."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Warning: Reference file not found: {file_path}")
        return "PhiX174_mock", "ATCGATCG" * 625  # Mock 5kb genome

    header = ""
    sequence = ""

    try:
        with open(file_path) as f:
            for line in f:
                if line.startswith('>'):
                    header = line[1:].strip()
                else:
                    sequence += line.strip().upper()

        if not sequence:
            print("Warning: No sequence found in reference file")
            return "PhiX174_mock", "ATCGATCG" * 625

    except Exception as e:
        print(f"Warning: Could not read reference file: {e}")
        return "PhiX174_mock", "ATCGATCG" * 625

    return header, sequence

def calculate_sequence_metrics(sequence: str) -> Dict[str, float]:
    """Calculate basic sequence metrics."""
    if not sequence:
        return {}

    length = len(sequence)
    gc_content = (sequence.count('G') + sequence.count('C')) / length * 100
    n_content = sequence.count('N') / length * 100

    # Calculate nucleotide frequencies
    nucleotides = ['A', 'T', 'C', 'G']
    frequencies = {nuc: sequence.count(nuc) / length * 100 for nuc in nucleotides}

    return {
        'length': length,
        'gc_content': gc_content,
        'n_content': n_content,
        'nucleotide_frequencies': frequencies
    }

def apply_design_filters(
    sequences: List[str],
    criteria: Dict[str, float]
) -> List[Tuple[int, str]]:
    """Apply filtering criteria to designed sequences."""
    filtered = []

    for i, seq in enumerate(sequences):
        metrics = calculate_sequence_metrics(seq)

        # Apply filters
        if (criteria['min_length'] <= metrics['length'] <= criteria['max_length'] and
            criteria['min_gc_content'] <= metrics['gc_content'] <= criteria['max_gc_content'] and
            metrics['n_content'] <= criteria['max_n_content']):

            filtered.append((i, seq))

    return filtered

def save_designed_genomes(
    sequences: List[str],
    file_path: Path,
    headers: Optional[List[str]] = None
) -> None:
    """Save designed genomes to FASTA file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        for i, seq in enumerate(sequences):
            if headers and i < len(headers):
                header = headers[i]
            else:
                header = f"designed_phage_genome_{i+1}"
            f.write(f">{header}\n{seq}\n")

# ==============================================================================
# Mock Design Functions (for CPU-only environment)
# ==============================================================================
def mock_design_genome_variant(
    reference_seq: str,
    config: Dict[str, Any],
    variant_id: int
) -> str:
    """Mock genome design that creates variants of reference."""
    # Set seed for reproducible variants
    random.seed(42 + variant_id)

    sequence = list(reference_seq.upper())
    mutation_rate = config['mutation_rate']
    nucleotides = ['A', 'T', 'C', 'G']

    # Apply mutations
    num_mutations = int(len(sequence) * mutation_rate)
    positions = random.sample(range(len(sequence)), min(num_mutations, len(sequence)))

    for pos in positions:
        original = sequence[pos]
        # Choose a different nucleotide
        choices = [n for n in nucleotides if n != original]
        sequence[pos] = random.choice(choices)

    # Ensure designed length
    target_length = config['design_length']
    current_length = len(sequence)

    if current_length < target_length:
        # Extend sequence
        extension_length = target_length - current_length
        for _ in range(extension_length):
            sequence.append(random.choice(nucleotides))
    elif current_length > target_length:
        # Truncate sequence
        sequence = sequence[:target_length]

    return ''.join(sequence)

def mock_design_genomes_batch(
    reference_seq: str,
    config: Dict[str, Any]
) -> List[str]:
    """Mock batch genome design."""
    print(f"[MOCK] Loading Evo2 model: {config['model_name']}")
    print(f"[MOCK] Model would require GPU/CUDA for actual generative inference")

    time.sleep(2)  # Simulate model loading and design time
    print(f"[MOCK] Designing {config['num_designs']} novel phage genomes...")

    designs = []
    for i in range(config['num_designs']):
        design = mock_design_genome_variant(reference_seq, config, i)
        designs.append(design)
        print(f"[MOCK] Generated design {i+1}: {len(design)} bp")

    return designs

# ==============================================================================
# Real Evo2 Functions (lazy loaded when GPU available)
# ==============================================================================
def get_evo2_model(model_name: str):
    """Lazy load Evo2 model to minimize startup time."""
    try:
        repo_path = Path(__file__).parent.parent / "repo"
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from evo2 import Evo2
        return Evo2(model_name)
    except Exception as e:
        print(f"Warning: Could not load Evo2 model: {e}")
        print("Falling back to mock mode")
        return None

def real_design_genomes_batch(
    reference_seq: str,
    config: Dict[str, Any]
) -> List[str]:
    """Real genome design using Evo2 generative capabilities."""
    model = get_evo2_model(config['model_name'])
    if model is None:
        print("Falling back to mock genome design")
        return mock_design_genomes_batch(reference_seq, config)

    print(f"Loading Evo2 model: {config['model_name']}")
    designs = []

    # Use reference as starting point for generation
    window_size = config['generation_params']['window_size']
    prompt = reference_seq[:window_size // 2]  # Use first half as prompt

    for i in range(config['num_designs']):
        print(f"Generating design {i+1}/{config['num_designs']}...")

        # Generate novel sequence
        design = model.generate(
            prompt,
            n_tokens=config['design_length'] - len(prompt),
            temperature=config['generation_params']['temperature'],
            top_k=config['generation_params']['top_k']
        )

        designs.append(design)

    return designs

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_phage_genome_design(
    reference_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for phage genome design.

    Args:
        reference_file: Path to reference genome FASTA file
        output_file: Path to save designed genomes FASTA file (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - designs: List of designed genome sequences
            - filtered_designs: List of designs that passed filters
            - reference_info: Information about reference genome
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_phage_genome_design("reference.fasta", output_file="designs.fasta")
        >>> print(f"Generated {len(result['designs'])} designs")
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}
    reference_file = Path(reference_file)

    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")

    # Set random seed for reproducibility in mock mode
    if config['mock_mode']:
        random.seed(42)

    # Load reference genome
    ref_header, ref_sequence = load_reference_genome(reference_file)
    ref_metrics = calculate_sequence_metrics(ref_sequence)

    print(f"Loaded reference genome: {ref_header}")
    print(f"Reference length: {ref_metrics['length']} bp")
    print(f"Reference GC content: {ref_metrics['gc_content']:.1f}%")

    # Design genomes
    if config['mock_mode']:
        designs = mock_design_genomes_batch(ref_sequence, config)
    else:
        designs = real_design_genomes_batch(ref_sequence, config)

    # Calculate metrics for all designs
    design_metrics = [calculate_sequence_metrics(design) for design in designs]

    # Apply filters
    print(f"\nApplying design filters...")
    filtered_results = apply_design_filters(designs, config['filter_criteria'])
    filtered_designs = [seq for _, seq in filtered_results]
    filtered_indices = [idx for idx, _ in filtered_results]

    print(f"Designs before filtering: {len(designs)}")
    print(f"Designs after filtering: {len(filtered_designs)}")

    # Create headers for designs
    headers = [f"designed_phage_{i+1}_from_{ref_header}" for i in range(len(designs))]
    filtered_headers = [headers[i] for i in filtered_indices]

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        if filtered_designs:
            save_designed_genomes(filtered_designs, output_path, filtered_headers)
        else:
            # Save all designs if none pass filters
            save_designed_genomes(designs, output_path, headers)
            print("Warning: No designs passed filters, saved all designs")

    return {
        "designs": designs,
        "filtered_designs": filtered_designs,
        "reference_info": {
            "header": ref_header,
            "sequence": ref_sequence,
            "metrics": ref_metrics
        },
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "config": config,
            "num_designs": len(designs),
            "num_filtered": len(filtered_designs),
            "design_metrics": design_metrics,
            "filtered_indices": filtered_indices
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
    parser.add_argument('--reference', type=str, required=True,
                       help='Reference genome FASTA file')
    parser.add_argument('--model', default='evo2_7b',
                       help='Evo2 model to use (default: evo2_7b)')
    parser.add_argument('--num-designs', type=int, default=5,
                       help='Number of designs to generate (default: 5)')
    parser.add_argument('--length', type=int, default=5000,
                       help='Target design length (default: 5000)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                       help='Mutation rate for mock designs (default: 0.1)')
    parser.add_argument('--temperature', type=float, default=1.2,
                       help='Generation temperature (default: 1.2)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output FASTA file for designed genomes')
    parser.add_argument('--config', '-c', type=str,
                       help='Config file (JSON)')
    parser.add_argument('--real-mode', action='store_true',
                       help='Use real Evo2 models (requires GPU)')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line args
    overrides = {
        'model_name': args.model,
        'num_designs': args.num_designs,
        'design_length': args.length,
        'mutation_rate': args.mutation_rate,
        'generation_params': {
            'temperature': args.temperature,
            'top_k': 8,
            'window_size': 1024
        },
        'mock_mode': not args.real_mode
    }

    # Print header
    print("=" * 60)
    if overrides['mock_mode']:
        print("MOCK VERSION - Evo2 Phage Genome Design Demo")
        print("This version runs without GPU/CUDA requirements")
        print("Use --real-mode for actual Evo2 models (requires GPU)")
    else:
        print("Evo2 Phage Genome Design")
        print("Using actual Evo2 models (requires GPU/CUDA)")
    print("=" * 60)

    # Run design
    result = run_phage_genome_design(
        reference_file=args.reference,
        output_file=args.output,
        config=config,
        **overrides
    )

    # Display results
    print("\n" + "=" * 50)
    print("GENOME DESIGN RESULTS:")
    print("=" * 50)

    ref_info = result['reference_info']
    print(f"Reference genome: {ref_info['header']}")
    print(f"Reference length: {ref_info['metrics']['length']} bp")
    print(f"Reference GC content: {ref_info['metrics']['gc_content']:.1f}%")

    print(f"\nDesign generation:")
    print(f"Total designs generated: {result['metadata']['num_designs']}")
    print(f"Designs passing filters: {result['metadata']['num_filtered']}")

    # Show metrics for designed genomes
    if result['metadata']['design_metrics']:
        metrics = result['metadata']['design_metrics']
        lengths = [m['length'] for m in metrics]
        gc_contents = [m['gc_content'] for m in metrics]

        print(f"\nDesigned genome statistics:")
        print(f"Length range: {min(lengths)} - {max(lengths)} bp")
        print(f"Mean length: {sum(lengths)/len(lengths):.1f} bp")
        print(f"GC content range: {min(gc_contents):.1f}% - {max(gc_contents):.1f}%")
        print(f"Mean GC content: {sum(gc_contents)/len(gc_contents):.1f}%")

    if result['output_file']:
        designs_saved = len(result['filtered_designs']) if result['filtered_designs'] else len(result['designs'])
        print(f"\n{designs_saved} designed genomes saved to: {result['output_file']}")

    print("\n" + "=" * 50)
    if overrides['mock_mode']:
        print("MOCK EXECUTION COMPLETED")
        print("For actual Evo2 inference, use --real-mode with GPU environment")
        print("Mock mode generates sequence variants with controlled mutations")
    else:
        print("EXECUTION COMPLETED")
    print("=" * 50)

    return result

if __name__ == '__main__':
    main()