# Step 6: MCP Tools Documentation

## Server Information
- **Server Name**: evo2
- **Version**: 1.0.0
- **Created Date**: 2025-12-24
- **Server Path**: `src/server.py`

## Overview

The evo2 MCP server provides comprehensive tools for DNA sequence analysis using the Evo2 foundation model. It offers both **synchronous APIs** for fast operations and **submit APIs** for long-running tasks, with full job management capabilities.

## Architecture

### API Types

1. **Synchronous API** - For fast operations (<10 minutes)
   - Direct function call, immediate response
   - Suitable for: small sequences, quick predictions, simple analysis

2. **Submit API** - For long-running tasks (>10 minutes) or batch processing
   - Submit job, get job_id, check status, retrieve results
   - Suitable for: large datasets, complex analysis, batch processing

### Job Management System

The server includes a comprehensive job management system with:
- **Job Queue**: Background execution of long-running tasks
- **Status Tracking**: Real-time progress monitoring
- **Result Retrieval**: Structured output when jobs complete
- **Log Access**: Execution logs for debugging and monitoring
- **Job Control**: Cancel running jobs if needed

## Job Management Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_job_status` | Check job progress and status | `job_id`: Job ID to check |
| `get_job_result` | Get completed job results | `job_id`: Job ID to retrieve |
| `get_job_log` | View job execution logs | `job_id`: Job ID, `tail`: Lines to show (default: 50) |
| `cancel_job` | Cancel running job | `job_id`: Job ID to cancel |
| `list_jobs` | List all jobs | `status`: Filter by status (optional) |

## Synchronous Tools (Fast Operations < 10 min)

| Tool | Description | Source Script | Est. Runtime | Input Limit |
|------|-------------|---------------|--------------|-------------|
| `generate_dna_sequences` | Generate short DNA sequences | `scripts/dna_generation.py` | ~30 sec | <1000 bases |
| `predict_variant_effects` | Predict variant pathogenicity | `scripts/variant_effect_prediction.py` | ~1-5 min | <100 variants |
| `score_sequences` | Score sequence quality | `scripts/sequence_scoring.py` | ~30 sec | <50 sequences |

### Tool Details

#### generate_dna_sequences
- **Description**: Generate DNA sequences from prompts using Evo2 models
- **Source Script**: `scripts/dna_generation.py`
- **Estimated Runtime**: ~30 seconds for short sequences

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| prompts | List[str] | Yes | - | DNA sequence prompts for generation |
| tokens | int | No | 100 | Number of tokens to generate per prompt |
| species | str | No | None | Species name for phylogenetic context |
| output_file | str | No | None | Path to save sequences as FASTA |

**Example:**
```
Use generate_dna_sequences with prompts ["ACGT", "ATCG"] and tokens 50
```

**Output:**
```json
{
  "status": "success",
  "sequences": ["ACGTACGTACGT...", "ATCGATCGATCG..."],
  "metadata": {
    "model_used": "evo2_7b",
    "generation_time": "2025-12-24T10:00:00",
    "statistics": [
      {"length": 50, "gc_content": 50.0},
      {"length": 50, "gc_content": 50.0}
    ]
  }
}
```

---

#### predict_variant_effects
- **Description**: Predict pathogenicity effects for small variant sets
- **Source Script**: `scripts/variant_effect_prediction.py`
- **Estimated Runtime**: ~1-5 minutes depending on variant count

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| variants_file | str | Yes | - | CSV/Excel file with variants (position, ref, alt) |
| reference_file | str | No | None | Reference sequence FASTA file |
| output_file | str | No | None | Path to save predictions as CSV |
| max_variants | int | No | 100 | Maximum variants to process (safety limit) |

**Example:**
```
Use predict_variant_effects with variants_file "examples/data/variants.csv"
```

---

#### score_sequences
- **Description**: Score DNA sequences for quality assessment
- **Source Script**: `scripts/sequence_scoring.py`
- **Estimated Runtime**: ~30 seconds for small sequence sets

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequences_file | str | Yes | - | FASTA file with DNA sequences |
| output_file | str | No | None | Path to save scores as CSV |
| normalize | bool | No | False | Whether to normalize scores |
| max_sequences | int | No | 50 | Maximum sequences to process |

**Example:**
```
Use score_sequences with sequences_file "examples/data/sample.fasta"
```

---

## Submit Tools (Long Operations > 10 min)

| Tool | Description | Source Script | Est. Runtime | Batch Support |
|------|-------------|---------------|--------------|---------------|
| `submit_dna_generation` | Large-scale DNA generation | `scripts/dna_generation.py` | >10 min | ✅ Yes |
| `submit_variant_effect_prediction` | Large variant set prediction | `scripts/variant_effect_prediction.py` | >10 min | ✅ Yes |
| `submit_sequence_embeddings` | Extract sequence embeddings | `scripts/sequence_embeddings.py` | >10 min | ✅ Yes |
| `submit_sequence_scoring` | Large-scale sequence scoring | `scripts/sequence_scoring.py` | >10 min | ✅ Yes |
| `submit_phage_genome_design` | Novel phage genome design | `scripts/phage_genome_design.py` | >10 min | ✅ Yes |

### Tool Details

#### submit_dna_generation
- **Description**: Submit DNA sequence generation for background processing
- **Source Script**: `scripts/dna_generation.py`
- **Estimated Runtime**: >10 minutes for large sequences
- **Supports Batch**: ✅ Yes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| prompts | List[str] | Yes | - | DNA sequence prompts for generation |
| tokens | int | No | 1000 | Number of tokens per prompt |
| species | str | No | None | Species for phylogenetic context |
| output_dir | str | No | None | Directory for outputs |
| job_name | str | No | auto | Custom job name |

**Example:**
```
Use submit_dna_generation with prompts ["ACGTACGT", "ATCGATCG"] and tokens 1000
→ Returns: {"job_id": "abc123", "status": "submitted"}
```

---

#### submit_variant_effect_prediction
- **Description**: Submit variant effect prediction for large variant sets
- **Source Script**: `scripts/variant_effect_prediction.py`
- **Estimated Runtime**: >10 minutes for large datasets
- **Supports Batch**: ✅ Yes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| variants_file | str | Yes | - | CSV/Excel file with variants |
| reference_file | str | No | None | Reference sequence FASTA |
| output_dir | str | No | None | Directory for outputs |
| job_name | str | No | auto | Custom job name |

**Example:**
```
Use submit_variant_effect_prediction with variants_file "large_variants.csv"
→ Returns: {"job_id": "def456", "status": "submitted"}
```

---

#### submit_sequence_embeddings
- **Description**: Extract deep learning embeddings from DNA sequences
- **Source Script**: `scripts/sequence_embeddings.py`
- **Estimated Runtime**: >10 minutes depending on sequence count
- **Supports Batch**: ✅ Yes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequences_file | str | Yes | - | FASTA file with sequences |
| layer | str | No | blocks.26 | Model layer for embeddings |
| output_dir | str | No | None | Directory for outputs |
| job_name | str | No | auto | Custom job name |

**Example:**
```
Use submit_sequence_embeddings with sequences_file "large_dataset.fasta"
→ Returns: {"job_id": "ghi789", "status": "submitted"}
```

---

#### submit_sequence_scoring
- **Description**: Large-scale sequence scoring for quality assessment
- **Source Script**: `scripts/sequence_scoring.py`
- **Estimated Runtime**: >10 minutes for large datasets
- **Supports Batch**: ✅ Yes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequences_file | str | Yes | - | FASTA file with sequences |
| normalize | bool | No | True | Whether to normalize scores |
| output_dir | str | No | None | Directory for outputs |
| job_name | str | No | auto | Custom job name |

**Example:**
```
Use submit_sequence_scoring with sequences_file "genome_sequences.fasta"
→ Returns: {"job_id": "jkl012", "status": "submitted"}
```

---

#### submit_phage_genome_design
- **Description**: Design novel bacteriophage genomes
- **Source Script**: `scripts/phage_genome_design.py`
- **Estimated Runtime**: >10 minutes depending on complexity
- **Supports Batch**: ✅ Yes

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| reference_file | str | Yes | - | Reference genome FASTA |
| num_designs | int | No | 5 | Number of designs to generate |
| target_length | int | No | 5000 | Target genome length |
| output_dir | str | No | None | Directory for outputs |
| job_name | str | No | auto | Custom job name |

**Example:**
```
Use submit_phage_genome_design with reference_file "phage_ref.fasta" and num_designs 10
→ Returns: {"job_id": "mno345", "status": "submitted"}
```

---

## Batch Processing Tools

### submit_batch_dna_generation
- **Description**: Process multiple prompt files in a single job
- **Use Case**: Large-scale DNA generation from multiple prompt sources
- **Supports**: Multiple prompt files, parallel processing

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| prompt_files | List[str] | Yes | - | Files containing prompts (one per line) |
| tokens | int | No | 500 | Tokens per prompt |
| output_dir | str | No | None | Directory for all outputs |
| job_name | str | No | auto | Custom job name |

### submit_batch_sequence_analysis
- **Description**: Comprehensive analysis (scoring + embeddings) for multiple files
- **Use Case**: Full analysis pipeline for multiple sequence datasets
- **Supports**: Scoring, embeddings, or combined analysis

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sequence_files | List[str] | Yes | - | FASTA files to analyze |
| analysis_type | str | No | all | "scoring", "embeddings", or "all" |
| output_dir | str | No | None | Directory for all outputs |
| job_name | str | No | auto | Custom job name |

---

## Workflow Examples

### Quick Analysis (Synchronous)
```
1. Use generate_dna_sequences with prompts ["ACGT"] and tokens 100
   → Returns results immediately (~30 seconds)

2. Use predict_variant_effects with variants_file "small_variants.csv"
   → Returns predictions immediately (~1 minute)

3. Use score_sequences with sequences_file "test_sequences.fasta"
   → Returns scores immediately (~30 seconds)
```

### Long-Running Task (Submit API)
```
1. Submit: Use submit_dna_generation with prompts ["ACGTACGT"] and tokens 1000
   → Returns: {"job_id": "abc123", "status": "submitted"}

2. Check: Use get_job_status with job_id "abc123"
   → Returns: {"status": "running", "started_at": "2025-12-24T10:05:00"}

3. Monitor: Use get_job_log with job_id "abc123" and tail 10
   → Returns: {"log_lines": ["Processing prompt 1...", "Generation 50% complete..."]}

4. Result: Use get_job_result with job_id "abc123" (when completed)
   → Returns: {"status": "success", "result": {...}}
```

### Batch Processing Workflow
```
1. Submit: Use submit_batch_sequence_analysis with sequence_files ["file1.fasta", "file2.fasta"]
   → Returns: {"job_id": "batch789", "status": "submitted"}

2. Monitor: Use get_job_status with job_id "batch789"
   → Track progress across all files

3. Results: Use get_job_result with job_id "batch789"
   → Get combined results from all processed files
```

### Error Handling
```
1. Submit job that fails:
   Use submit_dna_generation with prompts ["INVALID_BASES"]

2. Check status:
   Use get_job_status with job_id "failed123"
   → Returns: {"status": "failed", "error": "Invalid DNA bases in prompt"}

3. View logs:
   Use get_job_log with job_id "failed123"
   → See detailed error information
```

---

## Job Status Codes

| Status | Description | Actions Available |
|--------|-------------|-------------------|
| `pending` | Job queued, not started | Check status, cancel |
| `running` | Job executing | Check status, view logs, cancel |
| `completed` | Job finished successfully | Get results, view logs |
| `failed` | Job failed with error | View logs, check error message |
| `cancelled` | Job was cancelled | View logs up to cancellation |

---

## Output Formats

### Synchronous Tool Outputs
```json
{
  "status": "success|error",
  "error": "Error message (if status is error)",
  "sequences|predictions|scores": "Result data",
  "metadata": {
    "model_used": "evo2_7b",
    "execution_time": "ISO timestamp",
    "statistics": "Processing statistics"
  }
}
```

### Submit Tool Outputs
```json
{
  "status": "submitted",
  "job_id": "abc123",
  "message": "Job submitted. Use get_job_status('abc123') to check progress."
}
```

### Job Status Outputs
```json
{
  "job_id": "abc123",
  "job_name": "dna_gen_2_prompts",
  "status": "completed",
  "submitted_at": "2025-12-24T10:00:00",
  "started_at": "2025-12-24T10:00:05",
  "completed_at": "2025-12-24T10:15:23"
}
```

### Job Result Outputs
```json
{
  "status": "success",
  "result": {
    "sequences": ["ACGTACGT...", "ATCGATCG..."],
    "metadata": {...},
    "output_files": ["results/sequences.fasta"]
  }
}
```

---

## File Locations

### Generated Files
```
jobs/
├── abc123/                    # Job ID directory
│   ├── metadata.json         # Job metadata
│   ├── job.log               # Execution logs
│   └── output.json           # Structured results
├── def456/
│   └── ...
```

### Script Outputs
```
results/                       # Default output directory
├── dna_sequences.fasta       # Generated sequences
├── variant_predictions.csv   # Variant predictions
├── sequence_scores.csv       # Sequence scores
├── sequence_embeddings.csv   # Extracted embeddings
└── designed_genomes.fasta    # Designed genomes
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

## Error Handling

### Common Errors
1. **File not found**: Check input file paths
2. **Invalid format**: Ensure CSV has required columns, FASTA is valid
3. **Memory errors**: Use submit API for large datasets
4. **Model loading errors**: Check GPU availability for real mode

### Error Response Format
```json
{
  "status": "error",
  "error": "Detailed error message",
  "error_type": "FileNotFoundError|ValueError|RuntimeError"
}
```

---

## Configuration

### Mock vs Real Mode
- **Mock Mode**: CPU-compatible, deterministic results, no GPU required
- **Real Mode**: Requires GPU, actual Evo2 model inference

### Model Configuration
All tools can be configured via config files in `configs/` directory:
- `dna_generation_config.json`
- `variant_effect_prediction_config.json`
- `sequence_embeddings_config.json`
- `sequence_scoring_config.json`
- `phage_genome_design_config.json`

---

## Testing and Examples

### Test Data Available
```
examples/data/
├── NC_001422_1.fna           # PhiX174 genome for testing
├── sample_variants.csv       # Example variants
├── test_sequences.fasta      # Test sequences
└── reference_genome.fasta    # Reference for phage design
```

### Quick Test Commands
```bash
# Test sync DNA generation
Use generate_dna_sequences with prompts ["ACGT"] and tokens 50

# Test submit API
Use submit_dna_generation with prompts ["ACGTACGT"] and tokens 200

# Check job status
Use get_job_status with job_id "your_job_id"
```

---

## Success Criteria Checklist

- [x] MCP server created at `src/server.py`
- [x] Job manager implemented for async operations
- [x] Sync tools created for fast operations (<10 min)
- [x] Submit tools created for long-running operations (>10 min)
- [x] Batch processing support for applicable tools
- [x] Job management tools working (status, result, log, cancel, list)
- [x] All tools have clear descriptions for LLM use
- [x] Error handling returns structured responses
- [x] Comprehensive documentation created
- [x] Examples and workflow guides provided

---

## Next Steps

1. **Test Server**: Run `fastmcp dev src/server.py` to test
2. **Integration**: Add to Claude Desktop configuration
3. **Production**: Deploy with proper environment setup
4. **Monitoring**: Use job management tools for production monitoring
5. **Optimization**: Scale based on usage patterns