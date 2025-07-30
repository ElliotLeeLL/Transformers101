import torch
print(torch.version.cuda)     # Should show CUDA version, e.g., '11.8'
print(torch.cuda.device_count())  # Should be ≥ 1
print(torch.cuda.get_device_name(0))  # Name of GPU if available