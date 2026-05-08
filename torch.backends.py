import torch

print(torch._dynamo.list_backends())
print(torch._inductor.list_mode_options())
