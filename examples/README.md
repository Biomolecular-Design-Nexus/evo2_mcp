# Evo2 MCP Examples

This directory contains standalone Python scripts demonstrating various use cases for the Evo2 DNA language model.

## Available Use Cases

### UC-001: DNA Sequence Generation
**Script**: `use_case_1_dna_generation.py`
**Description**: Generate DNA sequences based on prompts using Evo2 models
**Example Usage**:
```bash
python examples/use_case_1_dna_generation.py --prompts ACGT ATCG --tokens 200
python examples/use_case_1_dna_generation.py --species "Homo sapiens" --tokens 500
```

### UC-002: Variant Effect Prediction
**Script**: `use_case_2_variant_effect_prediction.py`
**Description**: Predict functional effects of genetic variants using likelihood scoring
**Example Usage**:
```bash
python examples/use_case_2_variant_effect_prediction.py \
    --variants examples/data/variants.csv \
    --reference examples/data/reference.fasta \
    --output results.csv
```

### UC-003: DNA Sequence Embeddings
**Script**: `use_case_3_sequence_embeddings.py`
**Description**: Extract DNA sequence embeddings for downstream machine learning tasks
**Example Usage**:
```bash
python examples/use_case_3_sequence_embeddings.py \
    --input examples/data/sequences_with_labels.csv \
    --bidirectional \
    --train-classifier \
    --output-embeddings embeddings.npy
```

### UC-004: Phage Genome Design
**Script**: `use_case_4_phage_genome_design.py`
**Description**: Generate and filter novel bacteriophage genome designs
**Example Usage**:
```bash
python examples/use_case_4_phage_genome_design.py \
    --num-genomes 50 \
    --target-length 5000 \
    --min-length 4500 \
    --max-length 5500 \
    --output-dir phage_results
```

### UC-005: Sequence Likelihood Scoring
**Script**: `use_case_5_sequence_scoring.py`
**Description**: Score DNA sequences for quality assessment and comparison
**Example Usage**:
```bash
python examples/use_case_5_sequence_scoring.py \
    --input examples/data/sequences.fasta \
    --output scores.csv \
    --rank \
    --analyze \
    --plot score_distribution.png
```

## Demo Data

The `data/` directory contains sample data files:

| File | Description | Used by |
|------|-------------|---------|
| `prompts.csv` | Sample DNA sequence prompts | UC-001, UC-003 |
| `41586_2018_461_MOESM3_ESM.xlsx` | BRCA1 variant dataset | UC-002 |
| `samplePositions.tsv` | Exon classifier sample data | UC-003 |
| `NC_001422_1.fna` | PhiX174 reference genome | UC-004, UC-005 |
| `NC_001422.1_Gprotein.fasta` | PhiX174 spike protein | UC-004 |
| `NC_001422.1_pseudocircular.gff` | PhiX174 annotations | UC-004 |

## Requirements

All scripts require the conda environment to be activated:
```bash
mamba activate ./env  # or: conda activate ./env
```

## Common Parameters

Most scripts support these common parameters:
- `--model`: Evo2 model to use (default: varies by use case)
- `--output`: Output file path
- `--help`: Show detailed help for each script

## Environment Notes

**Current Status**: The environment has some dependency conflicts that need to be resolved:
- Transformer Engine installation requires CUDA development headers
- FastMCP has library compatibility issues
- PyTorch currently installed in CPU-only mode

**Recommended Setup**: Use the conda environment for basic functionality, and consider using Docker or a dedicated GPU environment for full GPU acceleration.