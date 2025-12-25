#!/usr/bin/env python3
"""
Script: sequence_embeddings.py
Description: DNA sequence embeddings extraction using Evo2 models

Original Use Case: examples/use_case_3_sequence_embeddings.py
Dependencies Removed: torch, sklearn imports with lazy loading, evo2 imports with lazy loading

Usage:
    python scripts/sequence_embeddings.py --sequences <file> --output <output_file>

Example:
    python scripts/sequence_embeddings.py --sequences examples/data/prompts.csv --output results/embeddings.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import numpy as np
import pandas as pd
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
    "layer_name": "blocks.26",
    "embedding_dim": 2560,  # Typical for evo2_7b
    "max_length": 2048,
    "batch_size": 8,
    "use_final_token": True,
    "mock_mode": True
}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def load_sequences_from_file(file_path: Union[str, Path]) -> List[str]:
    """Load DNA sequences from various file formats."""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return create_mock_sequences()

    try:
        if str(file_path).endswith(('.csv', '.tsv')):
            # Try to load from CSV
            df = pd.read_csv(file_path)
            # Look for sequence columns
            seq_cols = ['sequence', 'seq', 'dna', 'Sequence', 'DNA']
            for col in seq_cols:
                if col in df.columns:
                    return df[col].dropna().tolist()

            # If no sequence column found, use first column
            if len(df.columns) > 0:
                return df.iloc[:, 0].dropna().tolist()

        elif str(file_path).endswith(('.fasta', '.fa', '.fna')):
            # Load from FASTA
            sequences = []
            current_seq = ""
            with open(file_path) as f:
                for line in f:
                    if line.startswith('>'):
                        if current_seq:
                            sequences.append(current_seq)
                        current_seq = ""
                    else:
                        current_seq += line.strip().upper()
                if current_seq:
                    sequences.append(current_seq)
            return sequences
        else:
            # Try to read as plain text (one sequence per line)
            with open(file_path) as f:
                return [line.strip().upper() for line in f if line.strip()]

    except Exception as e:
        print(f"Warning: Could not read file: {e}")
        return create_mock_sequences()

def create_mock_sequences() -> List[str]:
    """Create mock DNA sequences for testing."""
    sequences = [
        "ATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTA",
        "TTAAGGCCTTAAGGCC",
        "CCCGGGAAATTTCCCG",
        "AGCTTAGCTTAAGCTT"
    ]
    return sequences

def save_embeddings_csv(
    embeddings: np.ndarray,
    sequences: List[str],
    file_path: Path,
    metadata: Optional[Dict] = None
) -> None:
    """Save embeddings to CSV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    data = {"sequence": sequences}

    # Add embedding dimensions
    for i in range(embeddings.shape[1]):
        data[f"embedding_{i}"] = embeddings[:, i]

    # Add metadata if provided
    if metadata:
        for key, values in metadata.items():
            if len(values) == len(sequences):
                data[key] = values

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

# ==============================================================================
# Mock Embedding Functions (for CPU-only environment)
# ==============================================================================
def mock_extract_embeddings(
    sequences: List[str],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Mock embedding extraction that simulates Evo2 embedding behavior.
    """
    print(f"[MOCK] Loading Evo2 model: {config['model_name']}")
    print(f"[MOCK] Model would require GPU/CUDA for actual inference")

    time.sleep(1)  # Simulate loading time
    print(f"[MOCK] Extracting embeddings for {len(sequences)} sequences...")

    # Create deterministic mock embeddings
    embeddings = []
    embedding_dim = config['embedding_dim']

    for i, seq in enumerate(sequences):
        # Set seed based on sequence for reproducible results
        random.seed(hash(seq) % 1000000)
        np.random.seed(hash(seq) % 1000000)

        # Create mock embedding based on sequence properties
        embedding = np.random.randn(embedding_dim) * 0.1  # Base random embedding

        # Add sequence-specific features
        gc_content = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
        length_factor = min(len(seq) / config['max_length'], 1.0)

        # Modify embedding based on sequence properties
        embedding[0] = gc_content  # First dimension encodes GC content
        embedding[1] = length_factor  # Second dimension encodes relative length
        embedding[2] = seq.count('N') / len(seq) if len(seq) > 0 else 0  # N content

        # Add some structure based on k-mers
        for j, kmer in enumerate(['AT', 'GC', 'CG', 'TA']):
            if j + 3 < embedding_dim:
                embedding[j + 3] = seq.count(kmer) / max(len(seq) - 1, 1)

        embeddings.append(embedding)

        if (i + 1) % 10 == 0:
            print(f"[MOCK] Processed {i + 1} sequences...")

    return np.array(embeddings)

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

def real_extract_embeddings(
    sequences: List[str],
    config: Dict[str, Any]
) -> np.ndarray:
    """Real embedding extraction using Evo2."""
    model = get_evo2_model(config['model_name'])
    if model is None:
        print("Falling back to mock embedding extraction")
        return mock_extract_embeddings(sequences, config)

    print(f"Loading Evo2 model: {config['model_name']}")
    embeddings = []

    for i, seq in enumerate(sequences):
        # Truncate sequence if too long
        if len(seq) > config['max_length']:
            seq = seq[:config['max_length']]

        # Extract embeddings
        embedding = model.get_embeddings(
            seq,
            layer=config['layer_name'],
            use_final_token=config['use_final_token']
        )
        embeddings.append(embedding)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} sequences...")

    return np.array(embeddings)

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_sequence_embeddings(
    sequences_file: Union[str, Path, List[str]],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for DNA sequence embeddings extraction.

    Args:
        sequences_file: Path to sequences file or list of sequences
        output_file: Path to save embeddings CSV file (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - embeddings: Numpy array of embeddings
            - sequences: List of input sequences
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_sequence_embeddings("sequences.csv", output_file="embeddings.csv")
        >>> print(result['embeddings'].shape)
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Load sequences
    if isinstance(sequences_file, list):
        sequences = sequences_file
    else:
        sequences_file = Path(sequences_file)
        sequences = load_sequences_from_file(sequences_file)

    if not sequences:
        raise ValueError("No sequences found")

    # Set random seed for reproducibility in mock mode
    if config['mock_mode']:
        random.seed(42)
        np.random.seed(42)

    print(f"Processing {len(sequences)} sequences...")

    # Extract embeddings
    if config['mock_mode']:
        embeddings = mock_extract_embeddings(sequences, config)
    else:
        embeddings = real_extract_embeddings(sequences, config)

    # Calculate statistics
    sequence_stats = []
    for seq in sequences:
        sequence_stats.append({
            'length': len(seq),
            'gc_content': (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0,
            'n_content': seq.count('N') / len(seq) if len(seq) > 0 else 0
        })

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        save_embeddings_csv(embeddings, sequences, output_path, {'length': [s['length'] for s in sequence_stats]})

    return {
        "embeddings": embeddings,
        "sequences": sequences,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "config": config,
            "embedding_shape": embeddings.shape,
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
                       help='File with DNA sequences (CSV, FASTA, or text)')
    parser.add_argument('--model', default='evo2_7b',
                       help='Evo2 model to use (default: evo2_7b)')
    parser.add_argument('--layer', default='blocks.26',
                       help='Model layer for embedding extraction (default: blocks.26)')
    parser.add_argument('--max-length', type=int, default=2048,
                       help='Maximum sequence length (default: 2048)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output CSV file for embeddings')
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
        'layer_name': args.layer,
        'max_length': args.max_length,
        'mock_mode': not args.real_mode
    }

    # Print header
    print("=" * 60)
    if overrides['mock_mode']:
        print("MOCK VERSION - Evo2 Sequence Embeddings Demo")
        print("This version runs without GPU/CUDA requirements")
        print("Use --real-mode for actual Evo2 models (requires GPU)")
    else:
        print("Evo2 Sequence Embeddings")
        print("Using actual Evo2 models (requires GPU/CUDA)")
    print("=" * 60)

    # Run embedding extraction
    result = run_sequence_embeddings(
        sequences_file=args.sequences,
        output_file=args.output,
        config=config,
        **overrides
    )

    # Display results
    print("\n" + "=" * 50)
    print("EMBEDDING EXTRACTION RESULTS:")
    print("=" * 50)

    embeddings = result['embeddings']
    print(f"Number of sequences: {len(result['sequences'])}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Show sequence statistics
    stats = result['metadata']['sequence_stats']
    lengths = [s['length'] for s in stats]
    gc_contents = [s['gc_content'] for s in stats]

    print(f"\nSequence statistics:")
    print(f"Length range: {min(lengths)} - {max(lengths)} bp")
    print(f"Mean length: {np.mean(lengths):.1f} bp")
    print(f"GC content range: {min(gc_contents):.2f} - {max(gc_contents):.2f}")
    print(f"Mean GC content: {np.mean(gc_contents):.2f}")

    # Show embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"Mean embedding norm: {np.linalg.norm(embeddings, axis=1).mean():.3f}")
    print(f"Embedding variance: {np.var(embeddings):.6f}")

    if result['output_file']:
        print(f"\nEmbeddings saved to: {result['output_file']}")

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