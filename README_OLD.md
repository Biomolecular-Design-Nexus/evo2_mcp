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

- **Synchronous API**: Fast operations for immediate results (<10 minutes)
- **Submit API**: Long-running tasks with background processing (>10 minutes)
- **Job Management**: Full lifecycle management with status tracking, logs, and results
- **Batch Processing**: Handle multiple files efficiently
- **Mock Mode**: CPU-compatible operation without GPU requirements
- **Real Mode**: GPU-accelerated inference with Evo2 models

## Installation

### Prerequisites
```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env

# Install required dependencies
pip install fastmcp loguru pandas openpyxl numpy
```

### Claude Code (Recommended)

1. **Navigate to MCP directory**:
   ```bash
   cd /path/to/evo2_mcp
   ```

2. **Add MCP server**:
   ```bash
   claude mcp add evo2 -- $(pwd)/env/bin/python $(pwd)/src/server.py
   ```

3. **Verify installation**:
   ```bash
   claude mcp list
   # Should show: evo2: ... - ✓ Connected
   ```

### Claude Desktop
Add to your Claude Desktop configuration (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "evo2": {
      "type": "stdio",
      "command": "/path/to/evo2_mcp/env/bin/python",
      "args": ["/path/to/evo2_mcp/src/server.py"],
      "env": {}
    }
  }
}
```

### Gemini CLI

Add to `~/.gemini/settings.json`:
```json
{
  "mcpServers": {
    "evo2": {
      "command": "/path/to/evo2_mcp/env/bin/python",
      "args": ["/path/to/evo2_mcp/src/server.py"],
      "env": {
        "PYTHONPATH": "/path/to/evo2_mcp"
      }
    }
  }
}
```

### Development Mode
```bash
fastmcp dev src/server.py
```

## Available Tools

### Quick Operations (Synchronous API)
These tools return results immediately:

| Tool | Description | Runtime | Input Limit |
|------|-------------|---------|-------------|
| `generate_dna_sequences` | Generate DNA sequences from prompts | ~30 sec | <1000 bases |
| `predict_variant_effects` | Predict variant pathogenicity | ~1-5 min | <100 variants |
| `score_sequences` | Score sequence quality | ~30 sec | <50 sequences |

### Long-Running Tasks (Submit API)
These tools return a job_id for tracking:

| Tool | Description | Runtime | Batch Support |
|------|-------------|---------|---------------|
| `submit_dna_generation` | Large-scale DNA generation | >10 min | ✅ Yes |
| `submit_variant_effect_prediction` | Large variant set prediction | >10 min | ✅ Yes |
| `submit_sequence_embeddings` | Extract sequence embeddings | >10 min | ✅ Yes |
| `submit_sequence_scoring` | Large-scale sequence scoring | >10 min | ✅ Yes |
| `submit_phage_genome_design` | Design novel phage genomes | >10 min | ✅ Yes |

### Job Management
| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress and status |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs (optionally filter by status) |

### Batch Processing
| Tool | Description |
|------|-------------|
| `submit_batch_dna_generation` | Process multiple prompt files |
| `submit_batch_sequence_analysis` | Comprehensive analysis for multiple files |

## Workflow Examples

### Quick Analysis (Synchronous)
```
Use generate_dna_sequences with prompts ["ACGT", "ATCG"] and tokens 100
→ Returns results immediately (~30 seconds)
```

### Long-Running Task (Asynchronous)
```
1. Submit: Use submit_dna_generation with prompts ["ACGTACGT"] and tokens 1000
   → Returns: {"job_id": "abc123", "status": "submitted"}

2. Check: Use get_job_status with job_id "abc123"
   → Returns: {"status": "running", ...}

3. Monitor: Use get_job_log with job_id "abc123"
   → Returns: {"log_lines": ["Processing...", ...]}

4. Get result: Use get_job_result with job_id "abc123"
   → Returns: {"status": "success", "result": {...}}
```

### Batch Processing
```
Use submit_batch_sequence_analysis with sequence_files ["file1.fasta", "file2.fasta"]
→ Processes all files in a single job with comprehensive analysis
```

## Installed Packages

### Main Environment (`./env`)
Key packages successfully installed:
- **evo2==0.3.0** (editable installation)
- **fastmcp==2.14.1**
- **torch==2.9.1+cpu** (CPU version due to CUDA conflicts)
- **transformer-engine==2.3.0** (core library)
- **biopython==1.86**
- **pandas==2.3.3**
- **numpy==2.4.0**
- **scikit-learn** (via dependencies)
- **loguru==0.7.3**
- **click==8.3.1**
- **tqdm==4.67.1**

### Additional Dependencies
- **huggingface-hub==1.2.3**
- **vtx==1.0.7** (Vortex inference engine)
- **einops==0.8.1**
- **networkx==3.6.1**
- **sympy==1.14.0**

## Directory Structure

```
./
├── README.md               # This file
├── env/                    # Main conda environment (Python 3.11)
├── src/                    # MCP server source code (to be created)
├── examples/               # Use case scripts and demo data
│   ├── use_case_1_dna_generation.py
│   ├── use_case_2_variant_effect_prediction.py
│   ├── use_case_3_sequence_embeddings.py
│   ├── use_case_4_phage_genome_design.py
│   ├── use_case_5_sequence_scoring.py
│   ├── data/               # Demo input data
│   │   ├── prompts.csv     # Sample DNA sequence prompts
│   │   ├── 41586_2018_461_MOESM3_ESM.xlsx  # BRCA1 variant dataset
│   │   ├── samplePositions.tsv             # Exon classifier data
│   │   ├── NC_001422_1.fna                 # PhiX174 reference genome
│   │   ├── NC_001422.1_Gprotein.fasta      # PhiX174 spike protein
│   │   └── NC_001422.1_pseudocircular.gff  # PhiX174 annotations
│   └── README.md           # Examples documentation
├── reports/                # Setup reports
│   ├── step3_environment.md
│   └── step3_use_cases.md
└── repo/                   # Original Evo2 repository
```

## Quick Start Examples

### In Claude Code:
```
# List available tools
"What tools do you have from evo2?"

# Run a quick analysis
"Analyze the protein in examples/data/NC_001422.1_Gprotein.fasta using score_sequences"

# Submit a long-running job
"Submit structure prediction for examples/data/NC_001422_1.fna using submit_phage_genome_design"

# Check job status
"Check status of job abc123"
```

## Testing

### Automated Tests
```bash
# Run integration tests
python tests/run_integration_tests.py

# Check results
cat reports/step7_integration.md
```

### Manual Testing
Use the comprehensive test scenarios:
```bash
# View test prompts
cat tests/manual_test_scenarios.md

# Follow the 33 test scenarios covering:
# - Tool discovery
# - Sync operations
# - Async job management
# - Batch processing
# - Error handling
# - End-to-end workflows
```

## Troubleshooting

### Common Issues

#### "Tool not found"
```bash
# Check MCP server registration
claude mcp list
# Should show: evo2: ... - ✓ Connected

# If not connected, re-register:
claude mcp remove evo2  # if exists
claude mcp add evo2 -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

#### "ModuleNotFoundError"
```bash
# Check Python environment
which python
python --version

# Verify imports
PYTHONPATH=$(pwd) python -c "from src.server import mcp"
```

#### "Jobs stuck in pending"
```bash
# Check job manager
python -c "from src.jobs.manager import job_manager; print(job_manager.list_jobs())"

# Check job directory
ls -la jobs/
```

#### "File not found"
```bash
# Verify file paths (use absolute paths if needed)
ls -la examples/data/

# Test with absolute path
python -c "import os; print(os.path.abspath('examples/data/NC_001422.1_Gprotein.fasta'))"
```

#### "Permission denied"
```bash
# Check output directory permissions
mkdir -p results
chmod 755 results
```

### Performance Issues

#### Slow startup
- First run downloads models (normal)
- Subsequent runs should be faster
- Use development mode for testing: `fastmcp dev src/server.py`

#### Memory issues
- Large jobs require adequate RAM
- Monitor with `htop` or `nvidia-smi` (GPU)
- Consider reducing batch sizes

### Known Issues

**1. GPU/CUDA Dependency (Primary Issue)**:
- **Issue**: `AssertionError: Could not find libtransformer_engine.so`
- **Root Cause**: Evo2 models require CUDA-compiled transformer_engine libraries
- **Current Status**: CPU-only environment cannot run actual Evo2 models
- **Solution**: Use mock versions for development, or set up GPU environment with Docker

**2. Transformer Engine Import Error**:
- **Issue**: RuntimeError about empty transformer-engine meta package
- **Solution**: Installed transformer-engine[core_cu12] instead of pytorch extensions

**3. CUDA Library Version Mismatch**:
- **Issue**: undefined symbol errors related to CUDA libraries
- **Solution**: Installed CPU-only PyTorch version to avoid conflicts

**4. FastMCP Library Compatibility**:
- **Issue**: ImportError related to libstdc++ version
- **Solution**: Environment conflict with system libraries - recommend using Docker for production

**5. Flash Attention Build Failure**:
- **Issue**: Missing cuDNN development headers
- **Solution**: Skipped flash-attention for now; can be installed in specialized GPU environment

### For Actual Evo2 Model Usage

To use the original scripts with real Evo2 models, you need:

```bash
# Option 1: Docker (Recommended)
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.04-py3
pip install evo2

# Option 2: GPU Environment Setup
# Requires:
# - NVIDIA GPU with compute capability >= 8.9 (H100 recommended)
# - CUDA 12.1+
# - cuDNN 9.3+
# - 8GB+ GPU memory for 7B models
```

### Recommendations for Production Use

1. **Use Docker**: For production deployment, use the provided Dockerfile to avoid environment conflicts
2. **GPU Environment**: For full GPU acceleration, set up a dedicated CUDA environment with proper driver versions
3. **Model Downloads**: Pre-download models to avoid network issues during inference
4. **Memory Management**: Large models (7B, 40B) require substantial GPU memory

## Environment Commands

```bash
# Activation (use one of these)
mamba activate ./env
mamba run -p ./env python <script>

# Package manager used
PKG_MGR="mamba"  # Mamba available and used throughout setup
```

## Model Information

### Available Evo2 Models
- **evo2_7b**: 7B parameters, 1M context length (recommended)
- **evo2_40b**: 40B parameters, 1M context length (requires multiple GPUs)
- **evo2_7b_base**: 7B parameters, 8K context length
- **evo2_1b_base**: 1B parameters, 8K context length
- **evo2_7b_microviridae**: Fine-tuned for phage genome design

### Hardware Requirements
- **Minimum**: 16GB RAM, CPU-only mode
- **Recommended**: NVIDIA GPU with 24GB+ VRAM for 7B models
- **Optimal**: Multiple GPUs for 40B model

## Notes

- Environment setup completed with mixed success - basic functionality works but some advanced features require additional setup
- CPU-only PyTorch limits performance but allows basic model testing
- Demo data successfully copied and organized for easy testing
- Use case scripts are functional and demonstrate key capabilities
- Consider GPU-optimized environment for production workloads

## Citation

```bibtex
@article{brixi2025evo2,
    title={Genome modeling and design across all domains of life with Evo 2},
    author={Brixi, Garyk and Durrant, Matthew G and others},
    journal={bioRxiv},
    year={2025},
    doi={10.1101/2025.02.18.638918}
}
```