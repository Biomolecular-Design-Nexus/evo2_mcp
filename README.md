# evo2 MCP

> A comprehensive MCP (Model Context Protocol) server for DNA sequence analysis using the Evo2 foundation model.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The evo2 MCP provides access to advanced DNA sequence analysis capabilities through the Evo2 foundation model. It enables DNA sequence generation, variant effect prediction, sequence scoring, embeddings extraction, and novel phage genome design through both direct script execution and MCP tool interfaces.

### Features
- **DNA Sequence Generation**: Generate realistic DNA sequences from prompts with phylogenetic context
- **Variant Effect Prediction**: Zero-shot pathogenicity prediction for genetic variants
- **Sequence Quality Scoring**: Likelihood-based assessment of DNA sequence quality
- **Sequence Embeddings**: Extract deep learning representations for downstream ML tasks
- **Phage Genome Design**: Design novel bacteriophage genomes with design constraints
- **Dual API**: Both synchronous (fast) and asynchronous (batch) processing
- **Mock Mode**: CPU-compatible fallback mode for testing without GPU requirements

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   ├── server.py           # MCP server
│   └── jobs/               # Job management system
├── scripts/
│   ├── dna_generation.py   # DNA sequence generation
│   ├── variant_effect_prediction.py # Variant pathogenicity prediction
│   ├── sequence_embeddings.py # Sequence embeddings extraction
│   ├── sequence_scoring.py # Sequence likelihood scoring
│   ├── phage_genome_design.py # Novel phage genome design
│   └── lib/                # Shared utilities
├── examples/
│   └── data/               # Demo data files
├── configs/                # JSON configuration files
└── repo/                   # Original Evo2 repository
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- Git

### Create Environment

Please follow the environment setup procedure as documented in `reports/step3_environment.md`:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/evo2_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install core dependencies
pip install pandas numpy tqdm pathlib typing

# Install MCP dependencies
pip install fastmcp loguru --ignore-installed

# Install additional data processing dependencies
pip install openpyxl biopython

# For GPU support (optional - requires CUDA environment)
# pip install torch torchvision torchaudio
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/dna_generation.py` | DNA sequence generation from prompts | See below |
| `scripts/variant_effect_prediction.py` | Variant pathogenicity prediction | See below |
| `scripts/sequence_scoring.py` | DNA sequence likelihood scoring | See below |
| `scripts/sequence_embeddings.py` | Sequence embeddings extraction | See below |
| `scripts/phage_genome_design.py` | Novel phage genome design | See below |

### Script Examples

#### DNA Generation

```bash
# Activate environment
mamba activate ./env

# Generate sequences from prompts
python scripts/dna_generation.py \
  --prompts ACGT,ATCG \
  --tokens 100 \
  --output results/sequences.fasta

# Generate with species context
python scripts/dna_generation.py \
  --prompts ACGT \
  --tokens 200 \
  --species "Homo sapiens" \
  --output results/human_sequences.fasta
```

**Parameters:**
- `--prompts`: DNA sequence prompts for generation (required)
- `--tokens`: Number of tokens to generate per prompt (default: 100)
- `--species`: Species name for phylogenetic context (optional)
- `--output`: Output FASTA file (default: stdout)

#### Variant Effect Prediction

```bash
python scripts/variant_effect_prediction.py \
  --variants examples/data/41586_2018_461_MOESM3_ESM.xlsx \
  --output results/variant_predictions.csv
```

**Parameters:**
- `--variants`: CSV/Excel file with variants (position, ref, alt columns) (required)
- `--reference`: Reference sequence FASTA file (optional)
- `--output`: Output CSV file (optional)
- `--max-variants`: Maximum variants to process for safety (default: 1000)

#### Sequence Scoring

```bash
python scripts/sequence_scoring.py \
  --sequences examples/data/NC_001422_1.fna \
  --normalize \
  --output results/sequence_scores.csv
```

**Parameters:**
- `--sequences`: FASTA file with DNA sequences (required)
- `--normalize`: Whether to normalize scores (flag)
- `--output`: Output CSV file (optional)
- `--max-sequences`: Maximum sequences to process (default: 1000)

#### Sequence Embeddings

```bash
python scripts/sequence_embeddings.py \
  --sequences examples/data/NC_001422_1.fna \
  --layer blocks.26 \
  --output results/embeddings.csv
```

**Parameters:**
- `--sequences`: FASTA file with DNA sequences (required)
- `--layer`: Model layer to extract embeddings from (default: blocks.26)
- `--output`: Output CSV file (optional)

#### Phage Genome Design

```bash
python scripts/phage_genome_design.py \
  --reference examples/data/NC_001422_1.fna \
  --num-designs 5 \
  --length 5000 \
  --output results/designed_genomes.fasta
```

**Parameters:**
- `--reference`: Reference genome FASTA file (required)
- `--num-designs`: Number of genome designs to generate (default: 5)
- `--length`: Target length for designed genomes (default: 5000)
- `--output`: Output FASTA file (optional)

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name evo2
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add evo2 -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "evo2": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/evo2_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/evo2_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from evo2?
```

#### Basic DNA Generation
```
Use generate_dna_sequences with prompts ["ACGT", "ATCG"] and tokens 50
```

#### Variant Analysis
```
Use predict_variant_effects with variants_file @examples/data/41586_2018_461_MOESM3_ESM.xlsx
```

#### Sequence Analysis
```
Use score_sequences with sequences_file @examples/data/NC_001422_1.fna
```

#### Long-Running Tasks (Submit API)
```
Submit DNA generation for large sequences:
Use submit_dna_generation with prompts ["ACGTACGT"] and tokens 1000

Then check the job status:
Use get_job_status with job_id "<returned_job_id>"

Get results when completed:
Use get_job_result with job_id "<returned_job_id>"
```

#### Batch Processing
```
Process multiple files in batch:
Use submit_batch_sequence_analysis with sequence_files ["@examples/data/NC_001422_1.fna", "@examples/data/NC_001422.1_Gprotein.fasta"]
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/NC_001422_1.fna` | Reference PhiX174 genome |
| `@examples/data/41586_2018_461_MOESM3_ESM.xlsx` | BRCA1 variant dataset |
| `@examples/data/prompts.csv` | Sample DNA sequence prompts |
| `@configs/dna_generation_config.json` | DNA generation configuration |
| `@results/` | Output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "evo2": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/evo2_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/evo2_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same syntax as Claude Code)
> What tools are available from evo2?
> Use generate_dna_sequences with prompts ["ACGT"] and tokens 100
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `generate_dna_sequences` | Generate DNA sequences from prompts | `prompts`, `tokens`, `species`, `output_file` |
| `predict_variant_effects` | Predict variant pathogenicity | `variants_file`, `reference_file`, `output_file`, `max_variants` |
| `score_sequences` | Score DNA sequence quality | `sequences_file`, `output_file`, `normalize`, `max_sequences` |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_dna_generation` | Large-scale DNA generation | `prompts`, `tokens`, `species`, `output_dir`, `job_name` |
| `submit_variant_effect_prediction` | Large variant set prediction | `variants_file`, `reference_file`, `output_dir`, `job_name` |
| `submit_sequence_embeddings` | Extract sequence embeddings | `sequences_file`, `layer`, `output_dir`, `job_name` |
| `submit_sequence_scoring` | Large-scale sequence scoring | `sequences_file`, `normalize`, `output_dir`, `job_name` |
| `submit_phage_genome_design` | Novel phage genome design | `reference_file`, `num_designs`, `target_length`, `output_dir`, `job_name` |

### Batch Processing Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_batch_dna_generation` | Process multiple prompt files | `prompt_files`, `tokens`, `output_dir`, `job_name` |
| `submit_batch_sequence_analysis` | Comprehensive analysis pipeline | `sequence_files`, `analysis_type`, `output_dir`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs |

---

## Examples

### Example 1: DNA Sequence Generation

**Goal:** Generate DNA sequences from prompts for synthetic biology applications

**Using Script:**
```bash
python scripts/dna_generation.py \
  --prompts ACGT,ATCG,GCTA \
  --tokens 100 \
  --output results/synthetic_sequences.fasta
```

**Using MCP (in Claude Code):**
```
Use generate_dna_sequences with prompts ["ACGT", "ATCG", "GCTA"] and tokens 100 to create synthetic DNA sequences
```

**Expected Output:**
- Generated DNA sequences (100 bases each)
- Metadata with GC content and sequence statistics
- Optional FASTA file output

### Example 2: Variant Effect Prediction

**Goal:** Predict pathogenicity of genetic variants from a clinical dataset

**Using Script:**
```bash
python scripts/variant_effect_prediction.py \
  --variants examples/data/41586_2018_461_MOESM3_ESM.xlsx \
  --output results/brca1_predictions.csv
```

**Using MCP (in Claude Code):**
```
Analyze the BRCA1 variant dataset using predict_variant_effects with variants_file @examples/data/41586_2018_461_MOESM3_ESM.xlsx
```

**Expected Output:**
- Pathogenicity scores for each variant
- Classification (benign/pathogenic)
- Statistical summary

### Example 3: Batch Analysis Pipeline

**Goal:** Comprehensive analysis of multiple DNA sequence files

**Using Script:**
```bash
# Process each file separately
for f in examples/data/*.fna; do
  python scripts/sequence_scoring.py --sequences "$f" --output "results/$(basename "$f" .fna)_scores.csv"
done
```

**Using MCP (in Claude Code):**
```
Submit comprehensive batch analysis for all FASTA files:
Use submit_batch_sequence_analysis with sequence_files ["@examples/data/NC_001422_1.fna", "@examples/data/NC_001422.1_Gprotein.fasta"] and analysis_type "all"
```

**Expected Output:**
- Combined scoring and embeddings for all files
- Consolidated analysis report
- Individual file results

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Size | Description | Use With |
|------|------|-------------|----------|
| `41586_2018_461_MOESM3_ESM.xlsx` | 2.3MB | BRCA1 variant dataset from Findlay et al. | Variant prediction |
| `NC_001422_1.fna` | 5.5KB | PhiX174 complete genome | Scoring, embeddings, design reference |
| `NC_001422.1_Gprotein.fasta` | 212B | PhiX174 spike protein sequence | Protein analysis |
| `NC_001422.1_pseudocircular.gff` | 7.2KB | PhiX174 genome annotations | Structural analysis |
| `prompts.csv` | 27KB | Sample DNA sequence prompts | DNA generation |
| `samplePositions.tsv` | 2MB | Exon classifier sample data | Embeddings, classification |

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `dna_generation_config.json` | DNA generation settings | model, generation params, output options |
| `variant_effect_prediction_config.json` | Variant prediction settings | window size, scoring thresholds, format options |
| `sequence_embeddings_config.json` | Embeddings extraction settings | layer selection, batch size, output format |
| `sequence_scoring_config.json` | Sequence scoring settings | window parameters, normalization, statistics |
| `phage_genome_design_config.json` | Phage design settings | design constraints, mutation rates, filters |

### Config Example

```json
{
  "model": {
    "name": "evo2_7b",
    "device": "cuda"
  },
  "generation": {
    "n_tokens": 100,
    "temperature": 1.0,
    "top_k": 4
  },
  "output": {
    "format": "fasta",
    "include_metadata": true
  }
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
pip install pandas numpy tqdm fastmcp loguru openpyxl biopython
```

**Problem:** Import errors
```bash
# Verify installation
python -c "import pandas, numpy, fastmcp; print('Dependencies OK')"
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove evo2
claude mcp add evo2 -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
python src/server.py --help

# Test tool discovery
python -c "
import sys
sys.path.insert(0, 'src')
from server import mcp
print(list(mcp.list_tools().keys()))
"
```

### Script Issues

**Problem:** Script execution errors
```bash
# Test in mock mode (no GPU required)
python scripts/dna_generation.py --prompts ACGT --tokens 10

# Check dependencies
python -c "import pandas, numpy; print('Core deps OK')"
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job status
python -c "
import sys; sys.path.insert(0, 'src')
from jobs.manager import job_manager
print(job_manager.list_jobs())
"
```

**Problem:** Job failed
```
Use get_job_log with job_id "<job_id>" and tail 100 to see error details
```

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test individual scripts
python scripts/dna_generation.py --prompts ACGT --tokens 10
python scripts/sequence_scoring.py --sequences examples/data/NC_001422_1.fna

# Test MCP server
python tests/run_integration_tests.py
```

### Starting Dev Server

```bash
# Run MCP server in development mode
fastmcp dev src/server.py

# Test with Claude Code
claude
```

---

## Performance Guidelines

### Choose Synchronous API When:
- Processing <100 variants
- Generating <1000 base sequences
- Analyzing <50 sequences
- Need immediate results
- Interactive analysis

### Choose Submit API When:
- Processing >100 variants
- Generating >1000 base sequences
- Analyzing >50 sequences
- Batch processing multiple files
- Background processing acceptable

### Choose Batch Processing When:
- Multiple input files
- Large-scale analysis
- Want to group related analyses
- Efficient resource utilization

---

## License

Based on the Evo2 repository. Please refer to the original repository for licensing information.

## Credits

Based on [Evo2: Genome Modeling and Design Across All Domains of Life](https://github.com/evo-design/evo)

## Support

For issues with the MCP server:
1. Check the troubleshooting section above
2. Review the integration test results in `reports/step7_integration.md`
3. Run manual tests using `tests/manual_test_scenarios.md`
4. Check job logs using the job management tools

For issues with the underlying Evo2 model, please refer to the original repository.