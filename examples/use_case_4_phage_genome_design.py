#!/usr/bin/env python3
"""
Use Case 4: Phage Genome Design with Evo2

Description: Generate and design novel bacteriophage genomes using Evo2.
This script demonstrates phage genome generation and filtering based on
architectural similarity and other design criteria.

Input: Design parameters and reference genome data
Output: Generated phage genome sequences
Complexity: Complex
Source: phage_gen/pipelines/genome_design_filtering_pipeline.py
Priority: High
Environment: ./env (with additional bioinformatics tools)
"""

import argparse
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

try:
    from evo2 import Evo2
    from Bio import SeqIO, SeqRecord, Seq
    from Bio.SeqUtils import GC
    import biotite.sequence as seq
    import biotite.sequence.io.fasta as fasta
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure the environment has evo2, biopython, and biotite")
    exit(1)


def generate_phage_genomes(
    model_name: str = 'evo2_7b_microviridae',
    num_genomes: int = 10,
    target_length: int = 5000,
    temperature: float = 1.0,
    species_tag: Optional[str] = None
) -> List[str]:
    """
    Generate phage genome sequences using Evo2.

    Args:
        model_name: Evo2 model to use (preferably microviridae fine-tuned)
        num_genomes: Number of genomes to generate
        target_length: Target genome length
        temperature: Sampling temperature
        species_tag: Optional species tag for targeted generation

    Returns:
        List of generated genome sequences
    """
    print(f"Loading Evo2 model: {model_name}")
    model = Evo2(model_name)

    # Prepare prompts
    if species_tag:
        prompts = [species_tag] * num_genomes
    else:
        # Use simple prompts for generic phage generation
        prompts = ['ATCG'] * num_genomes

    print(f"Generating {num_genomes} phage genomes...")
    generations = model.generate(
        prompts,
        n_tokens=target_length,
        temperature=temperature
    )

    return generations.sequences


def calculate_basic_stats(sequence: str) -> Dict:
    """Calculate basic sequence statistics."""
    return {
        'length': len(sequence),
        'gc_content': GC(sequence),
        'a_content': sequence.count('A') / len(sequence) * 100,
        't_content': sequence.count('T') / len(sequence) * 100,
        'g_content': sequence.count('G') / len(sequence) * 100,
        'c_content': sequence.count('C') / len(sequence) * 100,
    }


def filter_by_length(sequences: List[str], min_length: int, max_length: int) -> List[Tuple[int, str]]:
    """Filter sequences by length."""
    filtered = []
    for i, seq in enumerate(sequences):
        if min_length <= len(seq) <= max_length:
            filtered.append((i, seq))
    return filtered


def filter_by_gc_content(sequences: List[str], min_gc: float, max_gc: float) -> List[Tuple[int, str]]:
    """Filter sequences by GC content."""
    filtered = []
    for i, seq in enumerate(sequences):
        gc = GC(seq)
        if min_gc <= gc <= max_gc:
            filtered.append((i, seq))
    return filtered


def check_orf_content(sequence: str, min_orf_length: int = 300) -> Dict:
    """
    Check for open reading frames in the sequence.

    Returns basic ORF statistics.
    """
    # Simple ORF detection - look for start/stop codons
    start_codons = ['ATG']
    stop_codons = ['TAA', 'TAG', 'TGA']

    orfs = []

    for frame in range(3):
        seq_frame = sequence[frame:]

        for i in range(0, len(seq_frame) - 2, 3):
            codon = seq_frame[i:i+3]

            if codon in start_codons:
                # Look for stop codon
                for j in range(i + 3, len(seq_frame) - 2, 3):
                    stop_codon = seq_frame[j:j+3]
                    if stop_codon in stop_codons:
                        orf_length = j - i
                        if orf_length >= min_orf_length:
                            orfs.append({
                                'start': frame + i,
                                'end': frame + j,
                                'length': orf_length,
                                'frame': frame
                            })
                        break

    return {
        'num_orfs': len(orfs),
        'total_orf_length': sum(orf['length'] for orf in orfs),
        'orfs': orfs
    }


def save_sequences_as_fasta(sequences: List[str], output_file: str, prefix: str = "generated_phage"):
    """Save sequences as FASTA file."""
    records = []
    for i, seq in enumerate(sequences):
        record = SeqRecord.SeqRecord(
            Seq.Seq(seq),
            id=f"{prefix}_{i+1}",
            description=f"Generated phage genome {i+1}"
        )
        records.append(record)

    SeqIO.write(records, output_file, "fasta")
    print(f"Sequences saved to {output_file}")


def analyze_generated_genomes(sequences: List[str]) -> pd.DataFrame:
    """Analyze generated genomes and return summary statistics."""
    data = []

    for i, seq in enumerate(sequences):
        stats = calculate_basic_stats(seq)
        orf_stats = check_orf_content(seq)

        row = {
            'genome_id': f"genome_{i+1}",
            'length': stats['length'],
            'gc_content': stats['gc_content'],
            'num_orfs': orf_stats['num_orfs'],
            'total_orf_length': orf_stats['total_orf_length'],
            'orf_coverage': orf_stats['total_orf_length'] / stats['length'] * 100
        }
        data.append(row)

    return pd.DataFrame(data)


def filter_genomes(
    sequences: List[str],
    min_length: int = 4000,
    max_length: int = 7000,
    min_gc: float = 30,
    max_gc: float = 70,
    min_orfs: int = 5
) -> List[str]:
    """Apply multiple filters to generated genomes."""
    print(f"Starting with {len(sequences)} genomes")

    # Filter by length
    filtered_indices = set(range(len(sequences)))

    length_filtered = {i for i, seq in enumerate(sequences)
                      if min_length <= len(seq) <= max_length}
    filtered_indices &= length_filtered
    print(f"After length filter: {len(filtered_indices)} genomes")

    # Filter by GC content
    gc_filtered = {i for i, seq in enumerate(sequences)
                  if min_gc <= GC(seq) <= max_gc}
    filtered_indices &= gc_filtered
    print(f"After GC filter: {len(filtered_indices)} genomes")

    # Filter by ORF content
    orf_filtered = {i for i, seq in enumerate(sequences)
                   if check_orf_content(seq)['num_orfs'] >= min_orfs}
    filtered_indices &= orf_filtered
    print(f"After ORF filter: {len(filtered_indices)} genomes")

    return [sequences[i] for i in sorted(filtered_indices)]


def main():
    parser = argparse.ArgumentParser(description="Design phage genomes with Evo2")
    parser.add_argument('--model', default='evo2_7b',
                       help='Evo2 model to use')
    parser.add_argument('--num-genomes', type=int, default=20,
                       help='Number of genomes to generate')
    parser.add_argument('--target-length', type=int, default=5000,
                       help='Target genome length')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--species-tag',
                       help='Species tag for targeted generation')

    # Filtering parameters
    parser.add_argument('--min-length', type=int, default=4000,
                       help='Minimum genome length')
    parser.add_argument('--max-length', type=int, default=7000,
                       help='Maximum genome length')
    parser.add_argument('--min-gc', type=float, default=30,
                       help='Minimum GC content (%)')
    parser.add_argument('--max-gc', type=float, default=70,
                       help='Maximum GC content (%)')
    parser.add_argument('--min-orfs', type=int, default=5,
                       help='Minimum number of ORFs')

    # Output options
    parser.add_argument('--output-dir', default='phage_design_output',
                       help='Output directory')
    parser.add_argument('--output-prefix', default='designed_phage',
                       help='Prefix for output files')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate genomes
    sequences = generate_phage_genomes(
        args.model, args.num_genomes, args.target_length,
        args.temperature, args.species_tag
    )

    # Analyze all generated genomes
    print("\nAnalyzing generated genomes...")
    all_stats = analyze_generated_genomes(sequences)
    all_stats.to_csv(output_dir / f"{args.output_prefix}_all_stats.csv", index=False)

    # Apply filters
    print("\nFiltering genomes...")
    filtered_sequences = filter_genomes(
        sequences, args.min_length, args.max_length,
        args.min_gc, args.max_gc, args.min_orfs
    )

    if filtered_sequences:
        # Analyze filtered genomes
        filtered_stats = analyze_generated_genomes(filtered_sequences)
        filtered_stats.to_csv(output_dir / f"{args.output_prefix}_filtered_stats.csv", index=False)

        # Save filtered sequences
        save_sequences_as_fasta(
            filtered_sequences,
            output_dir / f"{args.output_prefix}_filtered.fasta",
            args.output_prefix
        )

        print(f"\nDesign Summary:")
        print(f"Generated: {len(sequences)} genomes")
        print(f"Filtered: {len(filtered_sequences)} genomes")
        print(f"Success rate: {len(filtered_sequences)/len(sequences)*100:.1f}%")

        print(f"\nFiltered Genome Statistics:")
        print(f"Length range: {filtered_stats['length'].min()}-{filtered_stats['length'].max()} bp")
        print(f"GC content range: {filtered_stats['gc_content'].min():.1f}-{filtered_stats['gc_content'].max():.1f}%")
        print(f"ORF count range: {filtered_stats['num_orfs'].min()}-{filtered_stats['num_orfs'].max()}")
    else:
        print("No genomes passed the filtering criteria!")

    # Save all sequences for reference
    save_sequences_as_fasta(
        sequences,
        output_dir / f"{args.output_prefix}_all.fasta",
        f"{args.output_prefix}_raw"
    )

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()