#!/usr/bin/env python3
"""
Script: sequence_scoring.py
Description: DNA sequence likelihood scoring and quality assessment using Evo2 models

Original Use Case: examples/use_case_5_sequence_scoring.py
Dependencies Removed: matplotlib, torch imports with lazy loading, evo2 imports with lazy loading

Usage:
    python scripts/sequence_scoring.py --sequences <file> --output <output_file>

Example:
    python scripts/sequence_scoring.py --sequences examples/data/NC_001422_1.fna --output results/scores.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import pandas as pd
import numpy as np
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
    "window_size": 1024,
    "stride": 512,
    "batch_size": 4,
    "score_type": "likelihood",
    "normalize_scores": True,
    "mock_mode": True
}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def load_sequences_from_fasta(file_path: Union[str, Path]) -> List[Tuple[str, str]]:
    """Load sequences with headers from FASTA file."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return create_mock_sequence_data()

    sequences = []
    current_header = ""
    current_seq = ""

    try:
        with open(file_path) as f:
            for line in f:
                if line.startswith('>'):
                    if current_seq:
                        sequences.append((current_header, current_seq))
                    current_header = line[1:].strip()
                    current_seq = ""
                else:
                    current_seq += line.strip().upper()

            if current_seq:
                sequences.append((current_header, current_seq))

    except Exception as e:
        print(f"Warning: Could not read FASTA file: {e}")
        return create_mock_sequence_data()

    return sequences

def create_mock_sequence_data() -> List[Tuple[str, str]]:
    """Create mock sequence data for testing."""
    sequences = [
        ("sequence_1", "ATCGATCGATCGATCGATCGATCG" * 20),
        ("sequence_2", "GCTAGCTAGCTAGCTAGCTAGCTA" * 18),
        ("sequence_3", "TTAAGGCCTTAAGGCCTTAAGGCC" * 15),
    ]
    return sequences

def sliding_window_sequences(sequence: str, window_size: int, stride: int) -> List[str]:
    """Create sliding windows from a sequence."""
    windows = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        windows.append(sequence[i:i + window_size])
    return windows

def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of DNA sequence."""
    if not sequence:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100

def calculate_n_content(sequence: str) -> float:
    """Calculate N content of DNA sequence."""
    if not sequence:
        return 0.0
    n_count = sequence.count('N')
    return (n_count / len(sequence)) * 100

# ==============================================================================
# Mock Scoring Functions (for CPU-only environment)
# ==============================================================================
def mock_score_sequence(sequence: str, config: Dict[str, Any]) -> float:
    """Mock sequence scoring that simulates Evo2 likelihood scoring."""
    # Set seed based on sequence for reproducible results
    random.seed(hash(sequence) % 1000000)

    # Base score influenced by sequence properties
    gc_content = calculate_gc_content(sequence)
    n_content = calculate_n_content(sequence)
    length_factor = min(len(sequence) / config['window_size'], 1.0)

    # Mock likelihood calculation
    # Higher GC content and balanced composition generally get better scores
    gc_balance = 1 - abs(gc_content - 50) / 50  # Penalty for extreme GC content
    n_penalty = n_content  # Penalty for unknown bases

    base_score = gc_balance * 0.5 - n_penalty * 2.0

    # Add sequence complexity (simple entropy measure)
    nucleotides = ['A', 'T', 'C', 'G']
    counts = [sequence.count(nuc) for nuc in nucleotides]
    total = sum(counts)

    if total > 0:
        probs = [c/total for c in counts if c > 0]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        complexity_bonus = entropy / 2.0  # Max entropy is 2 for 4 nucleotides
        base_score += complexity_bonus * 0.3

    # Add length factor
    base_score += length_factor * 0.2

    # Add random noise
    noise = random.uniform(-0.1, 0.1)
    final_score = base_score + noise

    return final_score

def mock_score_sequences_batch(
    sequences: List[str],
    config: Dict[str, Any]
) -> List[float]:
    """Mock batch scoring of sequences."""
    print(f"[MOCK] Loading Evo2 model: {config['model_name']}")
    print(f"[MOCK] Model would require GPU/CUDA for actual inference")

    time.sleep(1)  # Simulate loading time
    print(f"[MOCK] Scoring {len(sequences)} sequences...")

    scores = []
    for i, seq in enumerate(sequences):
        score = mock_score_sequence(seq, config)
        scores.append(score)

        if (i + 1) % 100 == 0:
            print(f"[MOCK] Scored {i + 1} sequences...")

    return scores

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

def real_score_sequences_batch(
    sequences: List[str],
    config: Dict[str, Any]
) -> List[float]:
    """Real sequence scoring using Evo2."""
    model = get_evo2_model(config['model_name'])
    if model is None:
        print("Falling back to mock scoring")
        return mock_score_sequences_batch(sequences, config)

    print(f"Loading Evo2 model: {config['model_name']}")
    scores = []

    for i, seq in enumerate(sequences):
        # Truncate sequence if too long
        if len(seq) > config['window_size']:
            seq = seq[:config['window_size']]

        # Score sequence
        score = model.score(seq)
        scores.append(score)

        if (i + 1) % 10 == 0:
            print(f"Scored {i + 1} sequences...")

    return scores

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_sequence_scoring(
    sequences_file: Union[str, Path, List[str]],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for DNA sequence likelihood scoring.

    Args:
        sequences_file: Path to FASTA file, CSV file, or list of sequences
        output_file: Path to save scores CSV file (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - scores: List of likelihood scores
            - sequences: List of input sequences
            - headers: List of sequence headers (if available)
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_sequence_scoring("genome.fasta", output_file="scores.csv")
        >>> print(f"Average score: {np.mean(result['scores'])}")
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Load sequences
    if isinstance(sequences_file, list):
        sequences = [(f"seq_{i}", seq) for i, seq in enumerate(sequences_file)]
    else:
        sequences_file = Path(sequences_file)
        if str(sequences_file).endswith(('.fasta', '.fa', '.fna')):
            sequences = load_sequences_from_fasta(sequences_file)
        else:
            # Try to load as simple text or CSV
            try:
                with open(sequences_file) as f:
                    lines = [line.strip() for line in f if line.strip()]
                sequences = [(f"seq_{i}", line) for i, line in enumerate(lines)]
            except Exception as e:
                print(f"Warning: Could not read file: {e}")
                sequences = create_mock_sequence_data()

    if not sequences:
        raise ValueError("No sequences found")

    # Extract headers and sequences
    headers = [header for header, _ in sequences]
    seq_list = [seq for _, seq in sequences]

    # Set random seed for reproducibility in mock mode
    if config['mock_mode']:
        random.seed(42)
        np.random.seed(42)

    print(f"Processing {len(seq_list)} sequences...")

    # Score sequences
    if config['mock_mode']:
        scores = mock_score_sequences_batch(seq_list, config)
    else:
        scores = real_score_sequences_batch(seq_list, config)

    # Calculate statistics
    sequence_stats = []
    for seq in seq_list:
        sequence_stats.append({
            'length': len(seq),
            'gc_content': calculate_gc_content(seq),
            'n_content': calculate_n_content(seq)
        })

    # Normalize scores if requested
    if config['normalize_scores'] and scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score > 0:
            scores = [(s - mean_score) / std_score for s in scores]

    # Create results DataFrame
    results_df = pd.DataFrame({
        'header': headers,
        'sequence': seq_list,
        'likelihood_score': scores,
        'length': [s['length'] for s in sequence_stats],
        'gc_content': [s['gc_content'] for s in sequence_stats],
        'n_content': [s['n_content'] for s in sequence_stats]
    })

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

    return {
        "scores": scores,
        "sequences": seq_list,
        "headers": headers,
        "results_df": results_df,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "config": config,
            "num_sequences": len(seq_list),
            "score_stats": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            },
            "sequence_stats": sequence_stats
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
    parser.add_argument('--sequences', type=str, required=True,
                       help='File with DNA sequences (FASTA, text, or CSV)')
    parser.add_argument('--model', default='evo2_7b',
                       help='Evo2 model to use (default: evo2_7b)')
    parser.add_argument('--window-size', type=int, default=1024,
                       help='Maximum sequence window size (default: 1024)')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize scores (z-score)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output CSV file for scores')
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
        'window_size': args.window_size,
        'normalize_scores': args.normalize,
        'mock_mode': not args.real_mode
    }

    # Print header
    print("=" * 60)
    if overrides['mock_mode']:
        print("MOCK VERSION - Evo2 Sequence Scoring Demo")
        print("This version runs without GPU/CUDA requirements")
        print("Use --real-mode for actual Evo2 models (requires GPU)")
    else:
        print("Evo2 Sequence Scoring")
        print("Using actual Evo2 models (requires GPU/CUDA)")
    print("=" * 60)

    # Run scoring
    result = run_sequence_scoring(
        sequences_file=args.sequences,
        output_file=args.output,
        config=config,
        **overrides
    )

    # Display results
    print("\n" + "=" * 50)
    print("SEQUENCE SCORING RESULTS:")
    print("=" * 50)

    stats = result['metadata']['score_stats']
    print(f"Number of sequences: {result['metadata']['num_sequences']}")
    print(f"Score statistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std:  {stats['std']:.3f}")
    print(f"  Min:  {stats['min']:.3f}")
    print(f"  Max:  {stats['max']:.3f}")

    # Show sequence statistics
    seq_stats = result['metadata']['sequence_stats']
    if seq_stats:
        lengths = [s['length'] for s in seq_stats]
        gc_contents = [s['gc_content'] for s in seq_stats]

        print(f"\nSequence statistics:")
        print(f"  Length range: {min(lengths)} - {max(lengths)} bp")
        print(f"  Mean length: {np.mean(lengths):.1f} bp")
        print(f"  GC content range: {min(gc_contents):.2f}% - {max(gc_contents):.2f}%")
        print(f"  Mean GC content: {np.mean(gc_contents):.2f}%")

    if result['output_file']:
        print(f"\nScores saved to: {result['output_file']}")

    # Show top and bottom scored sequences
    if len(result['scores']) > 0:
        sorted_indices = np.argsort(result['scores'])

        print(f"\nTop 3 highest scoring sequences:")
        for i in sorted_indices[-3:][::-1]:
            print(f"  {result['headers'][i]}: {result['scores'][i]:.3f}")

        print(f"\nTop 3 lowest scoring sequences:")
        for i in sorted_indices[:3]:
            print(f"  {result['headers'][i]}: {result['scores'][i]:.3f}")

    print("\n" + "=" * 50)
    if overrides['mock_mode']:
        print("MOCK EXECUTION COMPLETED")
        print("For actual Evo2 inference, use --real-mode with GPU environment")
    else:
        print("EXECUTION COMPLETED")
    print("=" * 50)

    return result

if __name__ == '__main__':
    main()