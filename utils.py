"""
Device utility for AMD GPU (DirectML) / CPU fallback.
Provides a single function to get the best available compute device.
"""

import torch

def get_device():
    """
    Returns the best available device for training.
    Priority: DirectML (AMD GPU) > CUDA (NVIDIA) > CPU
    """
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"[Device] Using DirectML (AMD GPU): {device}")
        return device
    except ImportError:
        pass

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
        return device

    device = torch.device("cpu")
    print("[Device] Using CPU (no GPU acceleration detected)")
    return device


def to_device(tensor_or_model, device):
    """Move a tensor or model to the specified device."""
    return tensor_or_model.to(device)
