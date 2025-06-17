import gc
import json
from collections import defaultdict

import torch


# TODO include settings for results recording, savepoints creation, manage backups, debugging files

class KVW_Informer():
    def __init__(self):
        try:
            with open('./utilities/kvw_config.json', 'r') as file:
                self.settings = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file kvw_config.json was not found.")
        except json.JSONDecodeError:
            print(f"Error: The file kvw_config.json is not a valid JSON file.")

    def log_gpu_memory(self, step_name: str, view=False, console=False) -> str | None:
        if view is True or console is True:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                info = f"{step_name}: {allocated:.4f}GB allocated, {reserved:.4f}GB reserved"
            else:
                info = f"{step_name}: No CUDA available"
            if view is True:
                print(f"{info}")
            elif console is True:
                return info

    def gpu_memory_analysis(self, view=False):
        if view is True:
            # Detailed GPU memory analysis
            print(torch.cuda.memory_summary())
            # List all tensors currently on GPU
            gpu_tensors = []
            for obj in gc.get_objects():
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    gpu_tensors.append((type(obj), obj.shape, obj.element_size() * obj.numel()))

            # Sort by size
            gpu_tensors.sort(key=lambda x: x[2], reverse=True)
            for i, (tensor_type, shape, size_bytes) in enumerate(gpu_tensors[:10]):  # Top 10
                print(f"{i + 1}. Shape: {shape}, Size: {size_bytes / 1e6:.1f}MB")

    def track_gpu_objects(self):
        """Safely track GPU objects without library loading issues"""
        gpu_objects = defaultdict(int)
        total_size = 0

        # Create a snapshot of current objects to avoid weak reference issues
        current_objects = list(gc.get_objects())

        for obj in current_objects:
            try:
                # First check if it's actually a tensor
                if isinstance(obj, torch.Tensor):
                    if obj.is_cuda:
                        obj_type = type(obj).__name__
                        size = obj.element_size() * obj.numel()
                        gpu_objects[f"\n{obj_type}_{obj.shape}"] += 1
                        total_size += size
            except (ReferenceError, RuntimeError, OSError, AttributeError):
                # Skip any problematic objects
                continue

        return dict(gpu_objects), total_size
