#!/usr/bin/env python3
"""
Use Case 2: Variant Effect Prediction with Evo2

Description: Predict the functional effect of genetic variants using Evo2 likelihood scoring.
This script demonstrates zero-shot variant effect prediction as shown in BRCA1 analysis.

Input: Reference and variant DNA sequences
Output: Delta likelihood scores and pathogenicity predictions
Complexity: Medium
Source: notebooks/brca1/brca1_zero_shot_vep.ipynb
Priority: High
Environment: ./env
"""

import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

try:
    from evo2 import Evo2
    from Bio import SeqIO
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure the environment has evo2, biopython, and scikit-learn")
    exit(1)


def create_variant_sequences(
    reference_seq: str,
    position: int,
    ref_base: str,
    alt_base: str,
    window_size: int = 8192
) -> Tuple[str, str]:
    """
    Create reference and variant sequences for a given SNV.

    Args:
        reference_seq: Full reference sequence
        position: 1-indexed position of the variant
        ref_base: Reference base
        alt_base: Alternative base
        window_size: Size of sequence window around variant

    Returns:
        Tuple of (reference_window, variant_window)
    """
    pos_0_indexed = position - 1

    # Create window around variant
    start = max(0, pos_0_indexed - window_size // 2)
    end = min(len(reference_seq), pos_0_indexed + window_size // 2)

    ref_window = reference_seq[start:end]
    snv_pos_in_window = min(window_size // 2, pos_0_indexed - start)

    # Create variant sequence
    variant_window = (ref_window[:snv_pos_in_window] +
                     alt_base +
                     ref_window[snv_pos_in_window + 1:])

    # Verify the variant was created correctly
    assert ref_window[snv_pos_in_window] == ref_base, f"Reference base mismatch"
    assert variant_window[snv_pos_in_window] == alt_base, f"Alternative base mismatch"
    assert len(variant_window) == len(ref_window), "Sequence length mismatch"

    return ref_window, variant_window


def predict_variant_effects(
    variants_df: pd.DataFrame,
    reference_seq: str,
    model_name: str = 'evo2_1b_base',
    window_size: int = 8192
) -> pd.DataFrame:
    """
    Predict variant effects using Evo2 likelihood scoring.

    Args:
        variants_df: DataFrame with columns 'position', 'ref', 'alt'
        reference_seq: Full reference sequence
        model_name: Evo2 model to use
        window_size: Sequence window size

    Returns:
        DataFrame with added delta_score column
    """
    print(f"Loading Evo2 model: {model_name}")
    model = Evo2(model_name)

    # Create sequences for all variants
    ref_sequences = []
    var_sequences = []
    ref_seq_to_index = {}
    ref_seq_indices = []

    print("Creating variant sequences...")
    for _, row in variants_df.iterrows():
        ref_seq, var_seq = create_variant_sequences(
            reference_seq, row['position'], row['ref'], row['alt'], window_size
        )

        # Store unique reference sequences to avoid redundant scoring
        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(ref_sequences)
            ref_sequences.append(ref_seq)

        ref_seq_indices.append(ref_seq_to_index[ref_seq])
        var_sequences.append(var_seq)

    ref_seq_indices = np.array(ref_seq_indices)

    # Score sequences with Evo2
    print(f"Scoring {len(ref_sequences)} reference sequences...")
    ref_scores = model.score_sequences(ref_sequences)

    print(f"Scoring {len(var_sequences)} variant sequences...")
    var_scores = model.score_sequences(var_sequences)

    # Calculate delta scores
    delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indices]

    # Add to dataframe
    result_df = variants_df.copy()
    result_df['delta_score'] = delta_scores
    result_df['pathogenicity_prediction'] = (delta_scores < 0).astype(int)

    return result_df


def load_variants_from_csv(file_path: str) -> pd.DataFrame:
    """Load variants from CSV file."""
    df = pd.read_csv(file_path)
    required_columns = ['position', 'ref', 'alt']

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def load_reference_sequence(file_path: str) -> str:
    """Load reference sequence from FASTA file."""
    with open(file_path, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            return str(record.seq)
    raise ValueError("No sequence found in reference file")


def main():
    parser = argparse.ArgumentParser(description="Predict variant effects with Evo2")
    parser.add_argument('--variants', required=True,
                       help='CSV file with variants (columns: position, ref, alt)')
    parser.add_argument('--reference', required=True,
                       help='FASTA file with reference sequence')
    parser.add_argument('--model', default='evo2_1b_base',
                       help='Evo2 model to use')
    parser.add_argument('--window-size', type=int, default=8192,
                       help='Sequence window size around variants')
    parser.add_argument('--output', required=True,
                       help='Output CSV file for results')
    parser.add_argument('--labels',
                       help='Column name with true labels for evaluation')

    args = parser.parse_args()

    # Load data
    print(f"Loading variants from {args.variants}")
    variants_df = load_variants_from_csv(args.variants)
    print(f"Loaded {len(variants_df)} variants")

    print(f"Loading reference sequence from {args.reference}")
    reference_seq = load_reference_sequence(args.reference)
    print(f"Reference sequence length: {len(reference_seq):,} bp")

    # Predict variant effects
    results_df = predict_variant_effects(
        variants_df, reference_seq, args.model, args.window_size
    )

    # Evaluate if labels are provided
    if args.labels and args.labels in results_df.columns:
        y_true = results_df[args.labels]
        y_scores = -results_df['delta_score']  # Negative because lower delta = more pathogenic

        auroc = roc_auc_score(y_true, y_scores)
        print(f"\nEvaluation Results:")
        print(f"AUROC: {auroc:.4f}")

        accuracy = ((results_df['pathogenicity_prediction'] == y_true).sum() /
                   len(results_df))
        print(f"Accuracy: {accuracy:.4f}")

    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Display summary
    print(f"\nSummary:")
    print(f"Total variants: {len(results_df)}")
    print(f"Predicted pathogenic: {results_df['pathogenicity_prediction'].sum()}")
    print(f"Predicted benign: {(1 - results_df['pathogenicity_prediction']).sum()}")
    print(f"Mean delta score: {results_df['delta_score'].mean():.6f}")


if __name__ == "__main__":
    main()