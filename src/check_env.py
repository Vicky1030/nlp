import sys
import torch

print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('PyTorch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('没有检测到 CUDA。请检查 NVIDIA Windows 驱动、WSL2 和 PyTorch CUDA 版本。')
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
x = torch.randn(64, 64, device='cuda')
print('CUDA calculation:', float((x @ x).mean()))
print('SUCCESS: GPU 可用于 PyTorch 计算')
