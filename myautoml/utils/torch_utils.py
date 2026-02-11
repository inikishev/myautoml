import copy
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import numpy as np

TORCH_INSTALLED = find_spec("torch") is not None
if TYPE_CHECKING:
    import torch

class CudaStatus:
    """
    Attributes:
        total_bytes: total CUDA memory.
        reserved_bytes: memory reserved by pytorch, not necessarily allocated
        allocated_bytes: memory allocated by pytorch
        free_bytes: total free memory (unreserved + unallocated)
    """
    def __init__(self, device_id: int | None):
        self.device_id = device_id
        self.torch_installed = TORCH_INSTALLED

        if TORCH_INSTALLED and device_id is not None:
            import torch
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:

                # Get total, reserved, and allocated memory
                self.total_bytes = torch.cuda.get_device_properties(device_id).total_memory
                self.reserved_bytes = torch.cuda.memory_reserved(device_id)
                self.allocated_bytes = torch.cuda.memory_allocated(device_id)
                self.free_bytes = (self.total_bytes - self.reserved_bytes) + (self.reserved_bytes - self.allocated_bytes)

            else:
                self.total_bytes = self.reserved_bytes = self.allocated_bytes = self.free_bytes = 0

        else:
            self.cuda_available = False
            self.total_bytes = self.reserved_bytes = self.allocated_bytes = self.free_bytes = 0


    def __repr__(self):
        if self.device_id is None: return "device_id is None, CUDA will not be used."
        if self.torch_installed is False: return "PyTorch is not installed."
        if self.cuda_available is False: return "CUDA is not available."
        return (
            f"CUDA is available. Memory status for device {self.device_id}:\n"
            f"  Total:      {self.total_bytes / (1024**3):.2f} GiB\n"
            f"  Allocated:  {self.allocated_bytes / (1024**3):.2f} GiB\n"
            f"  Reserved:   {self.reserved_bytes / (1024**3):.2f} GiB\n"
            f"  Free:       {self.free_bytes / (1024**3):.2f} GiB"
        )


def copy_state_dict(state: dict[str, Any], device=None) -> dict[str, Any]:
    """clones tensors and ndarrays, recursively copies dicts, deepcopies everything else, also moves to device if it is not None"""
    import torch
    c = state.copy()
    for k,v in state.items():
        if isinstance(v, torch.Tensor):
            if device is not None: v = v.to(device)
            c[k] = v.clone()
        if isinstance(v, np.ndarray): c[k] = v.copy()
        elif isinstance(v, dict): c[k] = copy_state_dict(v)
        else:
            if isinstance(v, torch.nn.Module) and device is not None: v = v.to(device)
            c[k] = copy.deepcopy(v)
    return c

def set_optimizer_lr_(opt: "torch.optim.Optimizer", lr: float):
    opt.defaults["lr"] = lr
    for g in opt.param_groups:
        g["lr"] = lr

