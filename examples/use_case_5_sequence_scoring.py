#!/usr/bin/env python3
"""
Use Case 5: DNA Sequence Likelihood Scoring with Evo2

Description: Score the likelihood of DNA sequences using Evo2 models.
This script demonstrates how to calculate likelihood scores for sequences,
which can be used for sequence quality assessment, design validation, or
comparative analysis.

Input: DNA sequences
Output: Likelihood scores and analysis
Complexity: Simple
Source: README.md examples and scoring functionality
Priority: Medium
Environment: ./env
"""

import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

try:
    from evo2 import Evo2
    from Bio import SeqIO
    from scipy import stats
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure the environment has evo2, biopython, and scipy")
    exit(1)


def score_sequences(
    sequences: List[str],
    model: Evo2,
    batch_size: int = 32
) -> List[float]:
    """
    Score a list of sequences using Evo2.

    Args:
        sequences: List of DNA sequences to score
        model: Loaded Evo2 model
        batch_size: Batch size for scoring

    Returns:
        List of likelihood scores
    """
    print(f"Scoring {len(sequences)} sequences...")

    # Score in batches to avoid memory issues
    all_scores = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        scores = model.score_sequences(batch)
        all_scores.extend(scores)

    return all_scores


def compare_sequence_versions(
    original_seqs: List[str],
    modified_seqs: List[str],
    model: Evo2
) -> pd.DataFrame:
    """
    Compare likelihood scores between original and modified sequences.

    Args:
        original_seqs: Original sequences
        modified_seqs: Modified versions of the sequences
        model: Evo2 model

    Returns:
        DataFrame with comparison results
    """
    print("Scoring original sequences...")
    original_scores = score_sequences(original_seqs, model)

    print("Scoring modified sequences...")
    modified_scores = score_sequences(modified_seqs, model)

    # Calculate differences
    score_differences = [mod - orig for mod, orig in zip(modified_scores, original_scores)]

    results_df = pd.DataFrame({
        'sequence_id': range(len(original_seqs)),
        'original_score': original_scores,
        'modified_score': modified_scores,
        'score_difference': score_differences,
        'improvement': [diff > 0 for diff in score_differences]
    })

    return results_df


def analyze_sequence_quality(sequences: List[str], model: Evo2) -> Dict:
    """
    Analyze the quality of sequences based on likelihood scores.

    Args:
        sequences: List of DNA sequences
        model: Evo2 model

    Returns:
        Dictionary with quality analysis results
    """
    scores = score_sequences(sequences, model)

    analysis = {
        'num_sequences': len(sequences),
        'mean_score': np.mean(scores),
        'median_score': np.median(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'scores': scores
    }

    # Identify outliers (sequences with very low scores)
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    lower_threshold = q1 - 1.5 * iqr

    outlier_indices = [i for i, score in enumerate(scores) if score < lower_threshold]
    analysis['outliers'] = outlier_indices
    analysis['num_outliers'] = len(outlier_indices)

    return analysis


def rank_sequences(sequences: List[str], model: Evo2) -> pd.DataFrame:
    """
    Rank sequences by their likelihood scores.

    Args:
        sequences: List of DNA sequences
        model: Evo2 model

    Returns:
        DataFrame with ranked sequences
    """
    scores = score_sequences(sequences, model)

    # Create DataFrame with sequences and scores
    df = pd.DataFrame({
        'sequence_id': range(len(sequences)),
        'sequence': sequences,
        'score': scores,
        'length': [len(seq) for seq in sequences]
    })

    # Rank by score (higher is better)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df


def load_sequences_from_file(file_path: str) -> List[str]:
    """Load sequences from FASTA or text file."""
    sequences = []

    if file_path.endswith(('.fasta', '.fa', '.fna')):
        # Load from FASTA
        with open(file_path, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequences.append(str(record.seq))
    else:
        # Load from text file (one sequence per line)
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    sequences.append(line.upper())

    return sequences


def plot_score_distribution(scores: List[float], output_file: str = None):
    """Plot distribution of likelihood scores."""
    plt.figure(figsize=(10, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Likelihood Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Likelihood Scores')

    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(scores, vert=True)
    plt.ylabel('Likelihood Score')
    plt.title('Box Plot of Scores')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Score DNA sequences with Evo2")
    parser.add_argument('--input', required=True,
                       help='Input file with sequences (FASTA or text)')
    parser.add_argument('--model', default='evo2_7b',
                       help='Evo2 model to use')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for scoring')
    parser.add_argument('--output',
                       help='Output CSV file for scores')
    parser.add_argument('--plot',
                       help='Output file for score distribution plot')
    parser.add_argument('--compare-to',
                       help='Second file to compare sequences against')
    parser.add_argument('--rank', action='store_true',
                       help='Rank sequences by score')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform quality analysis')

    args = parser.parse_args()

    # Load sequences
    print(f"Loading sequences from {args.input}")
    sequences = load_sequences_from_file(args.input)
    print(f"Loaded {len(sequences)} sequences")

    if not sequences:
        print("No sequences found in input file!")
        return

    # Load model
    print(f"Loading Evo2 model: {args.model}")
    model = Evo2(args.model)

    # Score sequences
    if args.compare_to:
        # Comparison mode
        print(f"Loading comparison sequences from {args.compare_to}")
        comparison_sequences = load_sequences_from_file(args.compare_to)

        if len(sequences) != len(comparison_sequences):
            print("Error: Input files must have the same number of sequences for comparison")
            return

        results_df = compare_sequence_versions(sequences, comparison_sequences, model)

        print(f"\nComparison Results:")
        print(f"Sequences improved: {results_df['improvement'].sum()}")
        print(f"Sequences degraded: {(~results_df['improvement']).sum()}")
        print(f"Mean score difference: {results_df['score_difference'].mean():.6f}")

        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")

    elif args.rank:
        # Ranking mode
        ranked_df = rank_sequences(sequences, model)

        print(f"\nTop 10 sequences by likelihood score:")
        print(ranked_df[['rank', 'sequence_id', 'score', 'length']].head(10))

        if args.output:
            ranked_df.to_csv(args.output, index=False)
            print(f"Ranked sequences saved to {args.output}")

    else:
        # Simple scoring mode
        scores = score_sequences(sequences, model)

        # Create results dataframe
        results_df = pd.DataFrame({
            'sequence_id': range(len(sequences)),
            'sequence': sequences,
            'score': scores,
            'length': [len(seq) for seq in sequences]
        })

        print(f"\nScoring Results:")
        print(f"Mean score: {np.mean(scores):.6f}")
        print(f"Median score: {np.median(scores):.6f}")
        print(f"Score range: {np.min(scores):.6f} to {np.max(scores):.6f}")

        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")

        # Optional quality analysis
        if args.analyze:
            analysis = analyze_sequence_quality(sequences, model)
            print(f"\nQuality Analysis:")
            print(f"Number of outliers: {analysis['num_outliers']}")
            if analysis['outliers']:
                print(f"Outlier sequence IDs: {analysis['outliers']}")

        # Optional plotting
        if args.plot:
            plot_score_distribution(scores, args.plot)


if __name__ == "__main__":
    main()