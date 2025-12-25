#!/usr/bin/env python3
"""
Use Case 3: DNA Sequence Embeddings with Evo2

Description: Extract and use DNA sequence embeddings from Evo2 for downstream tasks.
This script demonstrates how to get embeddings from different layers and use them
for classification tasks like exon prediction.

Input: DNA sequences
Output: Sequence embeddings and optional classification results
Complexity: Medium
Source: notebooks/exon_classifier/exon_classifier.ipynb
Priority: High
Environment: ./env
"""

import argparse
import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple

try:
    from evo2 import Evo2
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure the environment has evo2, scikit-learn")
    exit(1)


def get_sequence_embeddings(
    sequences: List[str],
    model: Evo2,
    layer_name: str = 'blocks.26',
    use_final_token: bool = True
) -> np.ndarray:
    """
    Extract embeddings for a list of sequences.

    Args:
        sequences: List of DNA sequences
        model: Loaded Evo2 model
        layer_name: Name of layer to extract embeddings from
        use_final_token: Whether to use final token embedding

    Returns:
        Array of embeddings with shape (n_sequences, embedding_dim)
    """
    embeddings_list = []

    print(f"Extracting embeddings from {layer_name} for {len(sequences)} sequences...")

    for seq in sequences:
        input_ids = torch.tensor(
            model.tokenizer.tokenize(seq),
            dtype=torch.int,
        ).unsqueeze(0).to(model.device)

        with torch.no_grad():
            _, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])

        if use_final_token:
            # Use embedding of the final token
            emb = embeddings[layer_name][0, -1, :].cpu().to(torch.float32).numpy()
        else:
            # Use mean pooling across all tokens
            emb = embeddings[layer_name][0, :, :].mean(dim=0).cpu().to(torch.float32).numpy()

        embeddings_list.append(emb)

    return np.array(embeddings_list)


def get_bidirectional_embeddings(
    sequences: List[str],
    model: Evo2,
    layer_name: str = 'blocks.26'
) -> np.ndarray:
    """
    Get bidirectional embeddings by processing sequences in both directions
    and concatenating the results.

    Args:
        sequences: List of DNA sequences
        model: Loaded Evo2 model
        layer_name: Layer name to extract embeddings from

    Returns:
        Array of concatenated forward and reverse embeddings
    """
    # Forward embeddings
    forward_embeddings = get_sequence_embeddings(sequences, model, layer_name)

    # Reverse complement sequences
    complement_map = str.maketrans('ATCG', 'TAGC')
    reverse_sequences = [seq.translate(complement_map)[::-1] for seq in sequences]

    # Reverse embeddings
    reverse_embeddings = get_sequence_embeddings(reverse_sequences, model, layer_name)

    # Concatenate
    return np.concatenate([forward_embeddings, reverse_embeddings], axis=1)


def train_classifier_on_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Train a classifier on sequence embeddings.

    Args:
        embeddings: Sequence embeddings
        labels: Binary labels
        test_size: Fraction of data to use for testing
        random_state: Random seed

    Returns:
        Dictionary with model and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)

    results = {
        'model': clf,
        'accuracy': accuracy,
        'auroc': auroc,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'classification_report': classification_report(y_test, y_pred)
    }

    return results


def load_sequences_from_csv(file_path: str) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    Load sequences and optional labels from CSV file.

    Expected format:
    sequence,label
    ATCGATCG,1
    GCTAGCTA,0
    """
    df = pd.read_csv(file_path)

    if 'sequence' not in df.columns:
        raise ValueError("CSV file must have a 'sequence' column")

    sequences = df['sequence'].tolist()
    labels = df['label'].values if 'label' in df.columns else None

    return sequences, labels


def save_embeddings(embeddings: np.ndarray, output_path: str):
    """Save embeddings to numpy file."""
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract DNA sequence embeddings with Evo2")
    parser.add_argument('--input', required=True,
                       help='Input CSV file with sequences (and optional labels)')
    parser.add_argument('--model', default='evo2_7b_base',
                       help='Evo2 model to use')
    parser.add_argument('--layer', default='blocks.26',
                       help='Layer name to extract embeddings from')
    parser.add_argument('--bidirectional', action='store_true',
                       help='Use bidirectional embeddings (forward + reverse)')
    parser.add_argument('--output-embeddings',
                       help='Output file to save embeddings (.npy)')
    parser.add_argument('--train-classifier', action='store_true',
                       help='Train a classifier on embeddings (requires labels)')
    parser.add_argument('--pooling', choices=['final', 'mean'], default='final',
                       help='Pooling strategy for embeddings')

    args = parser.parse_args()

    # Load sequences and labels
    print(f"Loading sequences from {args.input}")
    sequences, labels = load_sequences_from_csv(args.input)
    print(f"Loaded {len(sequences)} sequences")

    if labels is not None:
        print(f"Labels available: {len(np.unique(labels))} classes")

    # Load model
    print(f"Loading Evo2 model: {args.model}")
    model = Evo2(args.model)

    # Extract embeddings
    if args.bidirectional:
        print("Extracting bidirectional embeddings...")
        embeddings = get_bidirectional_embeddings(sequences, model, args.layer)
    else:
        print("Extracting unidirectional embeddings...")
        use_final_token = (args.pooling == 'final')
        embeddings = get_sequence_embeddings(
            sequences, model, args.layer, use_final_token
        )

    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings if requested
    if args.output_embeddings:
        save_embeddings(embeddings, args.output_embeddings)

    # Train classifier if requested and labels available
    if args.train_classifier and labels is not None:
        print("\nTraining classifier on embeddings...")
        results = train_classifier_on_embeddings(embeddings, labels)

        print(f"\nClassification Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"AUROC: {results['auroc']:.4f}")
        print(f"\nDetailed Report:")
        print(results['classification_report'])

    elif args.train_classifier and labels is None:
        print("Warning: --train-classifier requires labels in input file")

    print("\nEmbedding extraction completed!")


if __name__ == "__main__":
    main()