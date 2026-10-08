"""Optional allocator limits for scoring on a shared GPU."""
import os


def configure_cuda_budget(variable="AUDIO_EVALS_CUDA_MEMORY_GIB"):
    value = os.environ.get(variable, os.environ.get("AUDIO_EVALS_CUDA_MEMORY_GIB"))
    if value is None:
        return
    import torch
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        budget = float(value) * 1024**3
        if not 0 < budget <= total:
            raise ValueError("AUDIO_EVALS_CUDA_MEMORY_GIB must be positive and fit on the GPU")
        torch.cuda.set_per_process_memory_fraction(budget / total, 0)
