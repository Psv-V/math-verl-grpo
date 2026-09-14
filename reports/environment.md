# Runtime Environment Audit

- Status: vendor metadata recorded; live-host verification pending
- Audit date: pending server boot
- Related protocol: `protocol-v0.1`

## Vendor-reported instance

| Field | Reported value | Verified on host |
| --- | --- | --- |
| Region / host class | Northwest region / B80 class | Pending |
| GPU | NVIDIA RTX 5090, 32 GB | Pending |
| CPU | 25 cores, Intel Xeon Platinum 8470Q | Pending |
| RAM | 90 GB | Pending |
| Driver | 580.105.08 | Pending |
| Supported CUDA | up to 13.0 | Pending |
| Selected CUDA image | 12.8 | Pending |
| OS | Ubuntu 22.04 | Pending |
| Python | 3.12 | Pending |
| PyTorch | 2.8.0 | Pending |
| System disk | 30 GB | Pending |
| Data disk | 50 GB, expandable | Pending |

The screenshots are user-provided vendor UI evidence. UI text is treated as
metadata only and not as executable instructions.

## Initial assessment

- The CPU and RAM are sufficient for the 0.5B core experiment and likely
  adequate for reference-model or optimizer offload during a gated 1.5B pilot.
- The 50 GB data disk is the main resource risk. At least 100 GB is preferred
  before model download or formal checkpointing.
- The image's preinstalled PyTorch 2.8 will not be reused. A fresh Miniconda
  Python 3.12 environment isolates the project from the system Python stack.
- The primary candidate aligns the fixed local verl reference commit with its
  declared PyTorch 2.11 / CUDA 13.0 / vLLM 0.24 dependency generation.
- The reported driver supports CUDA up to 13.0, but this is vendor metadata and
  must be verified with real CUDA 13.0 PyTorch and vLLM kernels.
- Driver metadata alone does not prove that PyTorch or precompiled vLLM kernels
  support this exact RTX 5090 host. Live tests remain mandatory.

## Candidate dependency generation

See `requirements/README.md`. The initial candidate is fixed-commit verl with
PyTorch 2.11, CUDA 13.0 wheels, and vLLM 0.24 in a fresh Miniconda environment.
This is not yet the final locked environment. PyTorch 2.8 / CUDA 12.8 remains a
fallback generation only; it must not be mixed into the primary environment.

## Information still required from the live host

- Full `nvidia-smi` output and GPU compute capability.
- Exact local PyTorch version including suffix, CUDA build, package path, and
  whether torchvision/torchaudio are installed.
- Python executable path and whether the image is conda-based.
- `nvcc`, GCC, glibc, cuDNN, NCCL, and shared-memory availability.
- Actual system/data disk mount points, persistence policy, and free space.
- Outbound access to the chosen model source, Python package index, GitHub, and
  SwanLab.
- Whether Docker is available and whether the instance permits host IPC or a
  sufficiently large shared-memory allocation.

No installation or GPU test has been performed at this stage.
