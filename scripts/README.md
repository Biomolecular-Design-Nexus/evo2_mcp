# MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported
2. **Self-Contained**: Functions inlined where possible, shared utilities in `lib/`
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping
5. **GPU/CPU Flexible**: All scripts support mock mode for CPU-only environments

## Scripts Overview

| Script | Description | Repo Dependent | Config | Mock Mode |
|--------|-------------|----------------|--------|-----------|
| `dna_generation.py` | Generate DNA sequences from prompts | Lazy load | ✅ | ✅ |
| `variant_effect_prediction.py` | Predict variant pathogenicity | Lazy load | ✅ | ✅ |
| `sequence_embeddings.py` | Extract sequence embeddings | Lazy load | ✅ | ✅ |
| `sequence_scoring.py` | Score sequence likelihood | Lazy load | ✅ | ✅ |
| `phage_genome_design.py` | Design novel phage genomes | Lazy load | ✅ | ✅ |

## Quick Start

```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env

# Run a script in mock mode (default)
python scripts/dna_generation.py --prompts ACGT ATCG --output results/sequences.fasta

# Run with actual Evo2 models (requires GPU)
python scripts/dna_generation.py --prompts ACGT ATCG --output results/sequences.fasta --real-mode

# Use custom configuration
python scripts/dna_generation.py --config configs/dna_generation_config.json --prompts ACGT
```

## Script Details

### 1. DNA Generation (`dna_generation.py`)

**Description**: Generate DNA sequences from prompts or species-specific contexts

**Main Function**: `run_dna_generation(prompts=None, species=None, output_file=None, config=None, **kwargs)`

**Usage**:
```bash
# Generate from prompts
python scripts/dna_generation.py --prompts ACGT ATCGATCG --tokens 100 --output results/sequences.fasta

# Species-specific generation
python scripts/dna_generation.py --species "Homo sapiens" --tokens 200 --output results/human_sequences.fasta

# With configuration
python scripts/dna_generation.py --config configs/dna_generation_config.json --prompts ACGT
```

**Config**: `configs/dna_generation_config.json`

### 2. Variant Effect Prediction (`variant_effect_prediction.py`)

**Description**: Predict pathogenicity of genetic variants using likelihood scoring

**Main Function**: `run_variant_effect_prediction(variants_file, reference_file=None, output_file=None, config=None, **kwargs)`

**Usage**:
```bash
# Predict variant effects
python scripts/variant_effect_prediction.py \
    --variants examples/data/41586_2018_461_MOESM3_ESM.xlsx \
    --reference examples/data/NC_001422_1.fna \
    --output results/variant_predictions.csv

# With custom threshold
python scripts/variant_effect_prediction.py \
    --variants variants.csv \
    --threshold -0.3 \
    --output predictions.csv
```

**Config**: `configs/variant_effect_prediction_config.json`

### 3. Sequence Embeddings (`sequence_embeddings.py`)

**Description**: Extract DNA sequence embeddings for downstream ML tasks

**Main Function**: `run_sequence_embeddings(sequences_file, output_file=None, config=None, **kwargs)`

**Usage**:
```bash
# Extract embeddings from FASTA
python scripts/sequence_embeddings.py \
    --sequences examples/data/NC_001422_1.fna \
    --output results/embeddings.csv

# Extract from CSV with specific layer
python scripts/sequence_embeddings.py \
    --sequences examples/data/prompts.csv \
    --layer blocks.20 \
    --output embeddings.csv
```

**Config**: `configs/sequence_embeddings_config.json`

### 4. Sequence Scoring (`sequence_scoring.py`)

**Description**: Score DNA sequences for likelihood and quality assessment

**Main Function**: `run_sequence_scoring(sequences_file, output_file=None, config=None, **kwargs)`

**Usage**:
```bash
# Score sequences
python scripts/sequence_scoring.py \
    --sequences examples/data/NC_001422_1.fna \
    --output results/scores.csv

# With normalization
python scripts/sequence_scoring.py \
    --sequences genome.fasta \
    --normalize \
    --output normalized_scores.csv
```

**Config**: `configs/sequence_scoring_config.json`

### 5. Phage Genome Design (`phage_genome_design.py`)

**Description**: Design novel bacteriophage genomes based on reference

**Main Function**: `run_phage_genome_design(reference_file, output_file=None, config=None, **kwargs)`

**Usage**:
```bash
# Design phage genomes
python scripts/phage_genome_design.py \
    --reference examples/data/NC_001422_1.fna \
    --num-designs 3 \
    --length 5000 \
    --output results/designed_genomes.fasta
```

**Config**: `configs/phage_genome_design_config.json`

## Shared Library

Common functions are organized in `scripts/lib/`:

### `bio_utils.py`
- `calculate_gc_content(sequence)`: Calculate GC content percentage
- `calculate_n_content(sequence)`: Calculate N content percentage
- `format_sequence_display(sequence, max_display=50)`: Format for display
- `reverse_complement(sequence)`: Get reverse complement
- `sliding_window(sequence, window_size, step_size=1)`: Generate windows
- `validate_dna_sequence(sequence)`: Check if valid DNA
- `get_sequence_stats(sequence)`: Comprehensive statistics

### `io_utils.py`
- `load_fasta_sequence(file_path)`: Load single sequence from FASTA
- `load_sequences_from_fasta(file_path)`: Load multiple sequences with headers
- `save_fasta(sequences, file_path, headers=None)`: Save to FASTA
- `save_csv(data, file_path)`: Save to CSV
- `load_sequences_from_file(file_path)`: Auto-detect format and load
- `get_file_format(file_path)`: Detect file format

**Usage Example**:
```python
from scripts.lib import calculate_gc_content, load_fasta_sequence

# Load sequence and calculate stats
sequence = load_fasta_sequence("genome.fasta")
gc_content = calculate_gc_content(sequence)
print(f"GC content: {gc_content:.1f}%")
```

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped as an MCP tool:

```python
# Example MCP tool wrapper
from scripts.dna_generation import run_dna_generation

@mcp.tool()
def generate_dna_sequences(
    prompts: List[str],
    tokens: int = 100,
    output_file: str = None
) -> dict:
    """Generate DNA sequences using Evo2."""
    return run_dna_generation(
        prompts=prompts,
        n_tokens=tokens,
        output_file=output_file
    )
```

## Mock Mode vs Real Mode

All scripts support both mock and real modes:

### Mock Mode (Default)
- ✅ **CPU-only**: Runs without GPU/CUDA
- ✅ **Fast**: Quick execution for testing
- ✅ **Deterministic**: Reproducible results
- ✅ **Educational**: Shows expected I/O patterns
- ❌ **Limited**: Simplified biological accuracy

### Real Mode (--real-mode flag)
- ❌ **GPU Required**: Needs CUDA environment
- ❌ **Slow**: Model loading and inference time
- ✅ **Accurate**: Uses actual Evo2 models
- ✅ **Production**: Real biological insights

**Switching to Real Mode**:
```bash
# Add --real-mode flag to any script
python scripts/dna_generation.py --prompts ACGT --real-mode
```

## Configuration Files

All scripts support JSON configuration files in `configs/`:

```bash
# Use configuration file
python scripts/dna_generation.py --config configs/dna_generation_config.json

# Override specific parameters
python scripts/dna_generation.py --config configs/dna_generation_config.json --tokens 200
```

**Example config structure**:
```json
{
  "_description": "Configuration for DNA generation",
  "model": {
    "name": "evo2_7b",
    "device": "cuda"
  },
  "generation": {
    "n_tokens": 100,
    "temperature": 1.0
  },
  "mock_mode": {
    "enabled": true,
    "seed": 42
  }
}
```

## Testing

All scripts have been tested in CPU-only environment:

```bash
# Test all scripts
python scripts/dna_generation.py --prompts ACGT ATCG --tokens 50 --output results/test_dna.fasta
python scripts/variant_effect_prediction.py --variants examples/data/41586_2018_461_MOESM3_ESM.xlsx --output results/test_variants.csv
python scripts/sequence_embeddings.py --sequences examples/data/prompts.csv --output results/test_embeddings.csv
python scripts/sequence_scoring.py --sequences examples/data/NC_001422_1.fna --output results/test_scores.csv
python scripts/phage_genome_design.py --reference examples/data/NC_001422_1.fna --output results/test_designs.fasta
```

## Dependencies

### Essential (always required)
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `argparse`: CLI interfaces
- `pathlib`: Path handling

### Optional (for real mode)
- `torch`: PyTorch for models
- `evo2`: Main model library (requires GPU)
- `sklearn`: ML utilities (sequence embeddings)

### Lazy Loaded (when needed)
- Scripts use lazy loading to minimize startup time
- Real Evo2 models only loaded when `--real-mode` is specified
- Repo dependencies isolated and loaded only when available

## File Structure

```
scripts/
├── lib/                          # Shared utilities
│   ├── __init__.py              # Library exports
│   ├── bio_utils.py             # Biological utilities
│   └── io_utils.py              # I/O utilities
├── dna_generation.py            # DNA sequence generation
├── variant_effect_prediction.py # Variant pathogenicity prediction
├── sequence_embeddings.py       # Sequence embedding extraction
├── sequence_scoring.py          # Sequence likelihood scoring
├── phage_genome_design.py       # Phage genome design
└── README.md                    # This file
```

## Notes

- **All scripts work independently**: No external dependencies beyond standard packages
- **Mock mode enables development**: Test MCP interfaces without GPU requirements
- **Real mode for production**: Use actual Evo2 models when GPU is available
- **Consistent interfaces**: All scripts follow same CLI patterns
- **MCP-ready**: Main functions designed for easy wrapping in MCP tools