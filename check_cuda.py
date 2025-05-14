import torch
from torch._C import _cuda_emptyCache
from torch.cuda.memory import empty_cache
from torch.types import Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print(torch.backends.cudnn.version())
print(device)
print(torch.cuda.max_memory_allocated(device))
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()