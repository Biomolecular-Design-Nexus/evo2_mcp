"""
I/O utility functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional
import pandas as pd

def load_fasta_sequence(file_path: Union[str, Path]) -> str:
    """Load single DNA sequence from FASTA file (concatenates all sequences)."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    sequence = ""
    with open(file_path) as f:
        for line in f:
            if not line.startswith('>'):
                sequence += line.strip().upper()

    if not sequence:
        raise ValueError("No sequence found in FASTA file")

    return sequence

def load_sequences_from_fasta(file_path: Union[str, Path]) -> List[Tuple[str, str]]:
    """Load sequences with headers from FASTA file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    sequences = []
    current_header = ""
    current_seq = ""

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

    if not sequences:
        raise ValueError("No sequences found in FASTA file")

    return sequences

def save_fasta(
    sequences: List[str],
    file_path: Union[str, Path],
    headers: Optional[List[str]] = None
) -> None:
    """Save sequences to FASTA file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        for i, seq in enumerate(sequences):
            if headers and i < len(headers):
                header = headers[i]
            else:
                header = f"sequence_{i+1}"
            f.write(f">{header}\n{seq}\n")

def save_csv(
    data: dict,
    file_path: Union[str, Path],
    index: bool = False
) -> None:
    """Save data to CSV file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=index)

def load_sequences_from_file(file_path: Union[str, Path]) -> List[str]:
    """Load DNA sequences from various file formats."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Handle different file formats
    if str(file_path).endswith(('.fasta', '.fa', '.fna')):
        # Load from FASTA
        sequences_with_headers = load_sequences_from_fasta(file_path)
        return [seq for _, seq in sequences_with_headers]

    elif str(file_path).endswith(('.csv', '.tsv')):
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
        else:
            raise ValueError("No sequence data found in CSV file")

    else:
        # Try to read as plain text (one sequence per line)
        with open(file_path) as f:
            sequences = [line.strip().upper() for line in f if line.strip()]

        if not sequences:
            raise ValueError("No sequences found in text file")

        return sequences

def get_file_format(file_path: Union[str, Path]) -> str:
    """Detect file format based on extension."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    format_map = {
        '.fasta': 'fasta',
        '.fa': 'fasta',
        '.fna': 'fasta',
        '.csv': 'csv',
        '.tsv': 'tsv',
        '.txt': 'text',
        '.xlsx': 'excel'
    }

    return format_map.get(suffix, 'unknown')