"""MCP Server for evo2

Provides both synchronous and asynchronous (submit) APIs for all tools.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("evo2")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def generate_dna_sequences(
    prompts: List[str],
    tokens: int = 100,
    species: Optional[str] = None,
    output_file: Optional[str] = None
) -> dict:
    """
    Generate DNA sequences from prompts using Evo2 models.

    Fast operation suitable for generating short sequences (<1000 bases).
    For large-scale generation, use submit_dna_generation.

    Args:
        prompts: List of DNA sequence prompts for generation
        tokens: Number of tokens to generate per prompt (default: 100)
        species: Optional species name for phylogenetic context
        output_file: Optional path to save sequences as FASTA

    Returns:
        Dictionary with generated sequences and metadata
    """
    try:
        from dna_generation import run_dna_generation

        result = run_dna_generation(
            prompts=prompts,
            n_tokens=tokens,
            species=species,
            output_file=output_file
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except Exception as e:
        logger.error(f"DNA generation failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def predict_variant_effects(
    variants_file: str,
    reference_file: Optional[str] = None,
    output_file: Optional[str] = None,
    max_variants: int = 100
) -> dict:
    """
    Predict variant pathogenicity effects for small variant sets.

    Fast operation suitable for <100 variants. For large variant sets,
    use submit_variant_effect_prediction.

    Args:
        variants_file: Path to CSV/Excel file with variants (position, ref, alt columns)
        reference_file: Optional path to reference sequence FASTA file
        output_file: Optional path to save predictions as CSV
        max_variants: Maximum number of variants to process (safety limit)

    Returns:
        Dictionary with variant predictions and pathogenicity scores
    """
    try:
        from variant_effect_prediction import run_variant_effect_prediction

        result = run_variant_effect_prediction(
            variants_file=variants_file,
            reference_file=reference_file,
            output_file=output_file,
            max_variants=max_variants
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Variant effect prediction failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def score_sequences(
    sequences_file: str,
    output_file: Optional[str] = None,
    normalize: bool = False,
    max_sequences: int = 50
) -> dict:
    """
    Score DNA sequences for quality assessment (small batches).

    Fast operation suitable for <50 sequences. For large-scale scoring,
    use submit_sequence_scoring.

    Args:
        sequences_file: Path to FASTA file with DNA sequences
        output_file: Optional path to save scores as CSV
        normalize: Whether to normalize scores
        max_sequences: Maximum number of sequences to process

    Returns:
        Dictionary with sequence likelihood scores and statistics
    """
    try:
        from sequence_scoring import run_sequence_scoring

        result = run_sequence_scoring(
            sequences_file=sequences_file,
            output_file=output_file,
            normalize=normalize,
            max_sequences=max_sequences
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except Exception as e:
        logger.error(f"Sequence scoring failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_dna_generation(
    prompts: List[str],
    tokens: int = 1000,
    species: Optional[str] = None,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit DNA sequence generation for background processing.

    This operation may take more than 10 minutes for large sequences.
    Use get_job_status() to monitor progress and get_job_result() to retrieve results.

    Args:
        prompts: List of DNA sequence prompts for generation
        tokens: Number of tokens to generate per prompt (default: 1000)
        species: Optional species name for phylogenetic context
        output_dir: Directory to save outputs
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "dna_generation.py")

    # Convert list to comma-separated string for CLI
    prompts_str = ",".join(prompts) if prompts else ""

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "prompts": prompts_str,
            "tokens": tokens,
            "species": species,
            "output-dir": output_dir
        },
        job_name=job_name or f"dna_gen_{len(prompts) if prompts else 0}_prompts"
    )

@mcp.tool()
def submit_variant_effect_prediction(
    variants_file: str,
    reference_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit variant effect prediction for large variant sets (background processing).

    This operation may take more than 10 minutes for large variant sets.
    Suitable for processing hundreds to thousands of variants.

    Args:
        variants_file: Path to CSV/Excel file with variants
        reference_file: Optional path to reference sequence FASTA file
        output_dir: Directory to save outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the prediction job
    """
    script_path = str(SCRIPTS_DIR / "variant_effect_prediction.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "variants": variants_file,
            "reference": reference_file,
            "output-dir": output_dir
        },
        job_name=job_name or "variant_prediction"
    )

@mcp.tool()
def submit_sequence_embeddings(
    sequences_file: str,
    layer: str = "blocks.26",
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit sequence embeddings extraction for background processing.

    This operation extracts deep learning embeddings from DNA sequences.
    Can take >10 minutes for large sequence sets.

    Args:
        sequences_file: Path to FASTA file with DNA sequences
        layer: Model layer to extract embeddings from (default: blocks.26)
        output_dir: Directory to save outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the embeddings extraction
    """
    script_path = str(SCRIPTS_DIR / "sequence_embeddings.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "sequences": sequences_file,
            "layer": layer,
            "output-dir": output_dir
        },
        job_name=job_name or "sequence_embeddings"
    )

@mcp.tool()
def submit_sequence_scoring(
    sequences_file: str,
    normalize: bool = True,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit large-scale sequence scoring for background processing.

    This operation scores hundreds to thousands of DNA sequences.
    Can take >10 minutes for large sequence sets.

    Args:
        sequences_file: Path to FASTA file with DNA sequences
        normalize: Whether to normalize scores
        output_dir: Directory to save outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the scoring job
    """
    script_path = str(SCRIPTS_DIR / "sequence_scoring.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "sequences": sequences_file,
            "normalize": normalize,
            "output-dir": output_dir
        },
        job_name=job_name or "sequence_scoring"
    )

@mcp.tool()
def submit_phage_genome_design(
    reference_file: str,
    num_designs: int = 5,
    target_length: int = 5000,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit novel phage genome design for background processing.

    This operation designs novel bacteriophage genomes based on a reference.
    Can take >10 minutes depending on design complexity.

    Args:
        reference_file: Path to reference genome FASTA file
        num_designs: Number of genome designs to generate (default: 5)
        target_length: Target length for designed genomes (default: 5000)
        output_dir: Directory to save outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the design job
    """
    script_path = str(SCRIPTS_DIR / "phage_genome_design.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "reference": reference_file,
            "num-designs": num_designs,
            "length": target_length,
            "output-dir": output_dir
        },
        job_name=job_name or f"phage_design_{num_designs}_genomes"
    )

# ==============================================================================
# Batch Processing Tools
# ==============================================================================

@mcp.tool()
def submit_batch_dna_generation(
    prompt_files: List[str],
    tokens: int = 500,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch DNA generation for multiple prompt files.

    Processes multiple prompt files in a single job. Suitable for:
    - Processing many prompt sets at once
    - Large-scale DNA generation
    - Parallel processing of independent prompts

    Args:
        prompt_files: List of files containing prompts (one per line)
        tokens: Number of tokens to generate per prompt
        output_dir: Directory to save all outputs
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch job
    """
    script_path = str(SCRIPTS_DIR / "dna_generation.py")

    # Convert list to comma-separated string for CLI
    files_str = ",".join(prompt_files)

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "prompt-files": files_str,
            "tokens": tokens,
            "output-dir": output_dir,
            "batch-mode": True
        },
        job_name=job_name or f"batch_dna_gen_{len(prompt_files)}_files"
    )

@mcp.tool()
def submit_batch_sequence_analysis(
    sequence_files: List[str],
    analysis_type: str = "all",
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch sequence analysis (scoring + embeddings) for multiple files.

    Processes multiple sequence files with comprehensive analysis.
    Runs both scoring and embeddings extraction for each file.

    Args:
        sequence_files: List of FASTA files to analyze
        analysis_type: Type of analysis - "scoring", "embeddings", or "all"
        output_dir: Directory to save all outputs
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch analysis
    """
    # For batch analysis, we'll use the scoring script as the driver
    script_path = str(SCRIPTS_DIR / "sequence_scoring.py")

    files_str = ",".join(sequence_files)

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "sequence-files": files_str,
            "analysis-type": analysis_type,
            "output-dir": output_dir,
            "batch-mode": True
        },
        job_name=job_name or f"batch_analysis_{len(sequence_files)}_files"
    )

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()