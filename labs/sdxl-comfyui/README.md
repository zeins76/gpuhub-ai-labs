# SDXL + ComfyUI Lab

This lab will benchmark a few SDXL workflows in ComfyUI on a single 24GB GPU:

- SDXL base only
- SDXL + refiner
- SDXL + 1× ControlNet

The idea is to measure:

- time per image,
- peak VRAM usage,
- and see what a 24GB GPU can realistically handle for SDXL workflows.

Right now this lab is also a work in progress. Logs, example graphs, and results will be added as I run more experiments.

## GPU sanity check (GPUHub – NVIDIA GeForce RTX 5090)

Before running any SDXL or ComfyUI benchmarks, I ran a small sanity check on the GPU using PyTorch:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
print("Total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

## Simple tensor test
x = torch.randn((4096, 4096), device="cuda")
y = torch.matmul(x, x)
print("Matmul done, tensor shape:", y.shape)


## Output on GPUHub (NVIDIA GeForce RTX 5090):

CUDA available: True
GPU name: NVIDIA GeForce RTX 5090
Total VRAM (GB): 31.36
Matmul done, tensor shape: torch.Size([4096, 4096])

This confirms that:

- CUDA is working correctly,
- the GPU has enough VRAM for heavier experiments (SDXL, LoRA, etc.),
- and the environment is ready for the actual labs in this repo.
