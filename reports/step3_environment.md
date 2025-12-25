# Step 3: Environment Setup Report

## Python Version Detection
- **Detected Python Version**: 3.11+ (from pyproject.toml: >=3.11,<3.13)
- **Strategy**: Single environment setup (Python >= 3.10)
- **Package Manager**: Mamba (available at /home/xux/miniforge3/condabin/mamba)

## Main MCP Environment
- **Location**: ./env
- **Python Version**: 3.11.14 (installed)
- **Purpose**: Main MCP server and Evo2 functionality

## Dependencies Installed

### Core MCP Dependencies
- loguru==0.7.3
- click==8.3.1
- pandas==2.3.3
- numpy==2.4.0
- tqdm==4.67.1
- fastmcp==2.14.1

### Evo2 Core Dependencies
- evo2==0.3.0 (editable installation from ./repo/evo2)
- biopython==1.86
- huggingface-hub==1.2.3
- vtx==1.0.7 (Vortex inference engine)
- einops==0.8.1

### PyTorch Stack
- torch==2.9.1+cpu (CPU-only version due to CUDA conflicts)
- torchvision==0.24.1+cpu
- torchaudio==2.9.1+cpu
- pillow==12.0.0

### Transformer Engine
- transformer-engine==2.3.0 (core library only)
- transformer_engine_cu12==2.3.0

### Additional Scientific Computing
- networkx==3.6.1
- sympy==1.14.0
- filelock==3.20.1
- fsspec==2025.12.0
- jinja2==3.1.6
- markupsafe==3.0.3

## Activation Commands
```bash
# Main MCP environment
mamba activate ./env
# or
mamba run -p ./env python <script>
```

## Verification Status

### ✅ Successfully Installed
- [x] Main environment (./env) created successfully
- [x] Evo2 package installed in editable mode
- [x] FastMCP installed (with some compatibility issues)
- [x] Basic scientific computing stack functional
- [x] Demo data copied and organized

### ⚠️ Partial Success / Issues
- [ ] **Transformer Engine PyTorch Extensions**: Failed to build due to missing cuDNN headers
- [ ] **Flash Attention**: Failed to build due to CUDA library mismatches
- [ ] **Full GPU Acceleration**: CUDA version mismatches prevent GPU usage
- [ ] **FastMCP Runtime**: Library compatibility issues with system libstdc++

### ❌ Not Completed
- [ ] Flash Attention installation
- [ ] GPU-accelerated PyTorch
- [ ] Full transformer engine with PyTorch extensions

## Installation Issues Encountered

### Issue 1: Transformer Engine Build Failure
**Problem**: Building transformer_engine_torch failed due to missing cuDNN development headers
**Error**: `fatal error: cudnn.h: No such file or directory`
**Solution**: Installed transformer-engine[core_cu12] instead of full pytorch extensions

### Issue 2: CUDA Library Version Mismatch
**Problem**: PyTorch CUDA libraries incompatible with system CUDA
**Error**: `undefined symbol: __nvJitLinkGetErrorLogSize_12_9`
**Solution**: Installed CPU-only PyTorch to avoid conflicts

### Issue 3: FastMCP Runtime Issues
**Problem**: Library compatibility issues at runtime
**Error**: `version 'CXXABI_1.3.15' not found`
**Solution**: Documented issue for Docker/container deployment

### Issue 4: Flash Attention Build Failure
**Problem**: Failed to build due to PyTorch library conflicts
**Error**: Torch import errors during build process
**Solution**: Skipped for now - can be added in specialized GPU environment

## Environment Configuration

### Environment Variables
No special environment variables required for basic functionality.

### GPU Configuration
Current setup uses CPU-only mode. For GPU acceleration:
- CUDA 12.1+ required
- cuDNN 9.3+ required
- Compute Capability 8.9+ for FP8 support

## Performance Notes

### Current Limitations
- **CPU-Only Mode**: Significantly slower inference than GPU
- **Memory Usage**: Large models (7B+) may be slow on CPU
- **Model Downloads**: First use requires downloading multi-GB models

### Recommended Optimizations
1. **Docker Deployment**: Use containerized environment for production
2. **GPU Environment**: Set up dedicated CUDA environment for full performance
3. **Model Caching**: Pre-download models to avoid runtime delays
4. **Memory Management**: Consider smaller models (1B) for CPU-only testing

## Directory Structure Created
```
./env/                     # Main conda environment
├── lib/python3.11/site-packages/
│   ├── evo2/             # Editable install
│   ├── fastmcp/          # MCP server framework
│   ├── torch/            # CPU-only PyTorch
│   └── ...
└── bin/python            # Python 3.11.14
```

## Validation Commands
```bash
# Test basic imports (some may fail due to issues above)
mamba run -p ./env python -c "import pandas, numpy, tqdm; print('Basic deps OK')"

# Test Evo2 import (will fail due to transformer engine issues)
# mamba run -p ./env python -c "import evo2; print('Evo2 OK')"

# Test individual components
mamba run -p ./env python -c "import torch; print('PyTorch OK')"
mamba run -p ./env python -c "import Bio; print('BioPython OK')"
```

## Next Steps for Production
1. **Create Dockerfile** with proper CUDA environment
2. **Set up GPU environment** with compatible CUDA/cuDNN versions
3. **Implement MCP server code** using extracted use cases
4. **Add comprehensive testing** for all use cases
5. **Documentation** for model selection and performance optimization