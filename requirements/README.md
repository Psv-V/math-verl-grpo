# Server dependency plan

These files describe the first compatibility candidate for the selected rental
server. The project will use a fresh Miniconda environment and will not reuse
the image's preinstalled PyTorch 2.8. They do not claim that the environment has
been validated: the server has not yet been booted or tested.

## Known platform information

The rental UI reports:

- NVIDIA RTX 5090, 32 GB VRAM
- 25 CPU cores, Intel Xeon Platinum 8470Q
- 90 GB host RAM
- NVIDIA driver 580.105.08
- Ubuntu 22.04
- Python 3.12
- CUDA 12.8
- preinstalled PyTorch 2.8.0
- 30 GB system disk and 50 GB expandable data disk

These values are vendor metadata, not measured evidence. They must be verified
on the running host before installation.

## Selected dependency generation

The local reference checkout at
`1252cc71aa5bd82e5604322064d69bfe6454c660` defines a runtime world based on
PyTorch 2.11, CUDA 13.0, and vLLM 0.24.0. Because the project will use a fresh
Miniconda environment, it can align with that fixed source generation instead
of preserving the image's PyTorch 2.8 packages.

The initial candidate therefore uses:

- verl commit `1252cc71aa5bd82e5604322064d69bfe6454c660`
- PyTorch 2.11.0 from the official CUDA 13.0 wheel index
- vLLM 0.24.0
- Transformers 5.9.0
- TensorDict 0.10.0
- TransferQueue commit `434f8c476b4be24bc087e6e95070e64efcc739f9`
- mbridge commit `641a5a01de71080b2200d10e369090e40c9a351c`

This mirrors the dependency generation declared by the fixed reference commit.
It remains a candidate until a real import, CUDA-kernel, vLLM generation,
Ray-worker, and verl smoke test pass. If the reported driver cannot execute the
CUDA 13.0 wheels reliably, the documented fallback is the older PyTorch 2.8 /
CUDA 12.8 generation; the two generations must never be mixed.

## Files

- `environment/server-cu130-torch211.yml` creates the isolated Python 3.12
  Miniconda environment.
- `constraints/server-cu130-torch211.txt` protects ABI-critical packages.
- `requirements/server-cu130-torch211.txt` contains direct runtime dependencies.
- `requirements/dev.txt` contains test and lint tools.

A fully transitive lock file will be generated from the actual Linux server
after successful resolution and testing. Generating that lock on macOS would
produce the wrong platform markers and cannot validate CUDA wheels.

## Installation guardrails

Do not install these files before capturing the original image state. The image
PyTorch remains useful evidence about the vendor environment even though the
project will not use it. At minimum, save the output of:

```bash
python -VV
python -m pip --version
python -m pip freeze
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.__file__)"
```

Create the project environment with Miniconda:

```bash
conda env create -f environment/server-cu130-torch211.yml
conda activate math-grpo
```

After activation, verify that `which python` points inside the `math-grpo`
environment before installing GPU packages.

Install the ABI-critical PyTorch layer from the official CUDA 13.0 index first:

```bash
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

Only after the basic CUDA import/forward/backward checks pass, install the main
runtime file under its constraint file. The exact installer command will be
confirmed on the live host; do not use `pip install -U` on individual framework
packages afterward.

Install the fixed verl source separately with dependencies disabled, because
the dependencies are controlled by this project's files:

```bash
python -m pip install --no-deps \
  "verl @ git+https://github.com/verl-project/verl.git@1252cc71aa5bd82e5604322064d69bfe6454c660"
```

Do not install `flash-attn` in the first pass. It is a compiled extension tied
to the exact PyTorch/CUDA ABI. The initial text-only Qwen configuration can use
PyTorch SDPA. After the basic stack passes, `flash-attn==2.8.3` may be installed
from the official verl wheelhouse as a separately verified optimization.

## Intended installation order

The commands are deliberately not finalized until the running image is
inspected. The safe logical order is:

1. Capture the platform's original package and hardware state.
2. Create the isolated `math-grpo` Miniconda environment.
3. Install and verify the PyTorch 2.11 CUDA 13.0 wheel layer.
4. Install the runtime requirements under the constraint file.
5. Install fixed-commit verl with `--no-deps`.
6. Install development requirements.
7. Test CUDA bf16 forward/backward.
8. Test vLLM with Qwen2.5-0.5B-Instruct.
9. Test Ray workers and the verl entry point.
10. Freeze the complete resolved environment and record hashes.

The fixed dependency set includes `torchcodec`; verify that the server image
provides FFmpeg shared libraries before the full import audit. This text-only
experiment does not decode media, but matching the fixed verl core dependency
set avoids import-time surprises.

No SwanLab API key belongs in any requirements file, shell script, Git commit, or
captured environment report.
