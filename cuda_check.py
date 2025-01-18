import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print(torch.backends.cudnn.version())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)