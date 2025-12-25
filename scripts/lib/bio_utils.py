"""
Biological sequence utility functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""

def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of DNA sequence."""
    if not sequence:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100

def calculate_n_content(sequence: str) -> float:
    """Calculate N content (unknown bases) of DNA sequence."""
    if not sequence:
        return 0.0
    n_count = sequence.count('N')
    return (n_count / len(sequence)) * 100

def calculate_at_content(sequence: str) -> float:
    """Calculate AT content of DNA sequence."""
    if not sequence:
        return 0.0
    at_count = sequence.count('A') + sequence.count('T')
    return (at_count / len(sequence)) * 100

def format_sequence_display(sequence: str, max_display: int = 50) -> str:
    """Format sequence for display with truncation."""
    if len(sequence) <= max_display:
        return sequence
    half = max_display // 2
    return f"{sequence[:half]}...{sequence[-half:]}"

def reverse_complement(sequence: str) -> str:
    """Calculate reverse complement of DNA sequence."""
    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement_map.get(base, base) for base in reversed(sequence.upper()))

def sliding_window(sequence: str, window_size: int, step_size: int = 1) -> list:
    """Generate sliding windows from a sequence."""
    windows = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        windows.append(sequence[i:i + window_size])
    return windows

def validate_dna_sequence(sequence: str) -> bool:
    """Check if sequence contains only valid DNA bases."""
    valid_bases = set('ATCGN')
    return all(base.upper() in valid_bases for base in sequence)

def get_sequence_stats(sequence: str) -> dict:
    """Get comprehensive statistics for a DNA sequence."""
    return {
        'length': len(sequence),
        'gc_content': calculate_gc_content(sequence),
        'at_content': calculate_at_content(sequence),
        'n_content': calculate_n_content(sequence),
        'is_valid_dna': validate_dna_sequence(sequence)
    }