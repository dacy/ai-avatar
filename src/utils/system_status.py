import torch
from typing import Dict

def get_system_status() -> Dict[str, str]:
    """Get system status information including PyTorch and CUDA details."""
    status = {
        "pytorch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "status": "operational"  # Default status
    }
    
    if torch.cuda.is_available():
        status.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0)
        })
    
    return status 