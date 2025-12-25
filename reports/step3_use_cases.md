# Step 3: Use Cases Report

## Scan Information
- **Scan Date**: 2025-12-23
- **Filter Applied**: DNA/RNA sequence generation, long context DNA modeling, DNA sequence embedding, mutation effect prediction, sequence design
- **Python Version**: 3.11.14
- **Environment Strategy**: Single environment (./env)
- **Repository**: Evo2 - Genome modeling and design across all domains of life

## Use Cases Identified

### UC-001: DNA Sequence Generation
- **Description**: Generate DNA sequences based on prompts using Evo2 language model
- **Script Path**: `examples/use_case_1_dna_generation.py`
- **Complexity**: Simple
- **Priority**: High
- **Environment**: `./env`
- **Source**: `notebooks/generation/generation_notebook.ipynb`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| prompts | list | DNA sequence prompts | --prompts |
| model_name | string | Evo2 model to use | --model |
| n_tokens | integer | Number of tokens to generate | --tokens |
| temperature | float | Sampling temperature | --temperature |
| species | string | Species for phylogenetic tagging | --species |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| sequences | list | Generated DNA sequences |
| fasta_file | file | Optional FASTA output file |

**Example Usage:**
```bash
python examples/use_case_1_dna_generation.py --prompts ACGT ATCG --tokens 200
python examples/use_case_1_dna_generation.py --species "Homo sapiens" --tokens 500
```

**Example Data**: `examples/data/prompts.csv`

---

### UC-002: Variant Effect Prediction
- **Description**: Zero-shot prediction of genetic variant pathogenicity using likelihood scoring
- **Script Path**: `examples/use_case_2_variant_effect_prediction.py`
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env`
- **Source**: `notebooks/brca1/brca1_zero_shot_vep.ipynb`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| variants_file | file | CSV with variants (position, ref, alt) | --variants |
| reference_file | file | FASTA reference sequence | --reference |
| model_name | string | Evo2 model to use | --model |
| window_size | integer | Sequence window size | --window-size |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| results_csv | file | Variants with delta scores and predictions |
| auroc | float | Performance metric if labels provided |

**Example Usage:**
```bash
python examples/use_case_2_variant_effect_prediction.py \
    --variants examples/data/variants.csv \
    --reference examples/data/reference.fasta \
    --output results.csv
```

**Example Data**: `examples/data/41586_2018_461_MOESM3_ESM.xlsx` (BRCA1 variant dataset)

---

### UC-003: DNA Sequence Embeddings
- **Description**: Extract DNA sequence embeddings from Evo2 for downstream ML tasks
- **Script Path**: `examples/use_case_3_sequence_embeddings.py`
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env`
- **Source**: `notebooks/exon_classifier/exon_classifier.ipynb`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| sequences_file | file | CSV with sequences and optional labels | --input |
| model_name | string | Evo2 model to use | --model |
| layer_name | string | Layer to extract embeddings from | --layer |
| bidirectional | flag | Use bidirectional embeddings | --bidirectional |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| embeddings_npy | file | Numpy array of embeddings |
| classifier_results | dict | Classification performance metrics |

**Example Usage:**
```bash
python examples/use_case_3_sequence_embeddings.py \
    --input examples/data/sequences.csv \
    --bidirectional \
    --train-classifier \
    --output-embeddings embeddings.npy
```

**Example Data**: `examples/data/samplePositions.tsv` (exon classifier sample data)

---

### UC-004: Phage Genome Design
- **Description**: Generate and filter novel bacteriophage genomes with design constraints
- **Script Path**: `examples/use_case_4_phage_genome_design.py`
- **Complexity**: Complex
- **Priority**: High
- **Environment**: `./env`
- **Source**: `phage_gen/pipelines/genome_design_filtering_pipeline.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| model_name | string | Evo2 model (preferably microviridae) | --model |
| num_genomes | integer | Number of genomes to generate | --num-genomes |
| target_length | integer | Target genome length | --target-length |
| filter_params | various | Length, GC, ORF filtering criteria | Multiple |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| filtered_genomes | file | FASTA file with filtered genomes |
| statistics_csv | file | Genome analysis statistics |
| design_report | text | Summary of design success rate |

**Example Usage:**
```bash
python examples/use_case_4_phage_genome_design.py \
    --num-genomes 50 \
    --target-length 5000 \
    --min-length 4500 \
    --max-length 5500 \
    --output-dir phage_results
```

**Example Data**: `examples/data/NC_001422_1.fna` (PhiX174 reference genome)

---

### UC-005: Sequence Likelihood Scoring
- **Description**: Score DNA sequences for quality assessment and comparison
- **Script Path**: `examples/use_case_5_sequence_scoring.py`
- **Complexity**: Simple
- **Priority**: Medium
- **Environment**: `./env`
- **Source**: README.md examples and evo2.score_sequences functionality

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| sequences_file | file | FASTA or text file with sequences | --input |
| model_name | string | Evo2 model to use | --model |
| comparison_file | file | Optional second file for comparison | --compare-to |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| scores_csv | file | Sequences with likelihood scores |
| ranked_csv | file | Sequences ranked by score |
| score_plot | image | Distribution of scores |

**Example Usage:**
```bash
python examples/use_case_5_sequence_scoring.py \
    --input examples/data/sequences.fasta \
    --output scores.csv \
    --rank \
    --analyze \
    --plot score_distribution.png
```

**Example Data**: Various FASTA files in `examples/data/`

---

## Summary

| Metric | Count |
|--------|-------|
| **Total Found** | 5 |
| **Scripts Created** | 5 |
| **High Priority** | 4 |
| **Medium Priority** | 1 |
| **Low Priority** | 0 |
| **Demo Data Copied** | ✅ |

## Use Case Categories

### By Complexity
- **Simple** (2): DNA Generation, Sequence Scoring
- **Medium** (2): Variant Effect Prediction, Sequence Embeddings
- **Complex** (1): Phage Genome Design

### By Application Domain
- **Sequence Generation** (2): Basic generation, Phage design
- **Sequence Analysis** (2): Variant effects, Likelihood scoring
- **Machine Learning** (1): Embeddings extraction

### By Data Requirements
- **Minimal Data** (2): Generation scripts (can run with simple prompts)
- **Reference Data** (1): Variant prediction (needs reference genome)
- **Labeled Data** (1): Embeddings (optional labels for classification)
- **Complex Data** (1): Phage design (multiple data types)

## Demo Data Index

| Source | Destination | Description |
|--------|-------------|-------------|
| `repo/evo2/test/data/prompts.csv` | `examples/data/prompts.csv` | Sample DNA sequence prompts for generation |
| `repo/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx` | `examples/data/41586_2018_461_MOESM3_ESM.xlsx` | BRCA1 variant dataset from Findlay et al. |
| `repo/evo2/notebooks/exon_classifier/samplePositions.tsv` | `examples/data/samplePositions.tsv` | Sample genomic positions for exon classification |
| `repo/evo2/phage_gen/data/NC_001422_1.fna` | `examples/data/NC_001422_1.fna` | PhiX174 reference genome |
| `repo/evo2/phage_gen/data/NC_001422.1_Gprotein.fasta` | `examples/data/NC_001422.1_Gprotein.fasta` | PhiX174 spike (G) protein sequence |
| `repo/evo2/phage_gen/data/NC_001422.1_pseudocircular.gff` | `examples/data/NC_001422.1_pseudocircular.gff` | PhiX174 pseudo-circularized annotations |

## Implementation Notes

### Script Features
- **Error Handling**: All scripts include proper error handling and informative messages
- **Argument Parsing**: Comprehensive CLI interfaces with help documentation
- **Modular Design**: Core functions separated from CLI code for reusability
- **Output Options**: Flexible output formats (CSV, FASTA, images)
- **Progress Tracking**: Progress bars and status updates for long operations

### Environment Dependencies
- All scripts designed to work with the `./env` conda environment
- Import error handling for missing dependencies
- Graceful degradation when optional packages unavailable
- Clear error messages guiding users to resolve issues

### Data Handling
- Support for standard bioinformatics formats (FASTA, CSV)
- Automatic format detection where possible
- Validation of input data structure and content
- Consistent error handling for malformed data

### Performance Considerations
- Batch processing for large datasets
- Memory-efficient implementations
- Progress tracking for long-running operations
- Configurable batch sizes and parameters

## Next Steps

### For MCP Integration
1. **Create MCP Server**: Integrate scripts into FastMCP server framework
2. **Define Tool Interfaces**: Map use cases to MCP tool definitions
3. **Add Streaming**: Implement streaming for long-running generations
4. **Error Handling**: Robust error handling for MCP tool calls

### For Production Deployment
1. **GPU Optimization**: Resolve CUDA environment for full performance
2. **Model Management**: Implement model caching and selection
3. **Scalability**: Add support for distributed inference
4. **Monitoring**: Add logging and performance metrics

### For User Experience
1. **Documentation**: Complete user guides with examples
2. **Testing**: Comprehensive test suite for all use cases
3. **Validation**: Input validation and sanitization
4. **Examples**: More diverse example datasets