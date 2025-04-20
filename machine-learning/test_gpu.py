# import torch

# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("Device count:", torch.cuda.device_count())
#     print("Device name:", torch.cuda.get_device_name(0))
#     tensor = torch.tensor([1.0, 2.0]).to('cuda')
#     print(tensor * 2)
# else:
#     print("CUDA is not available on this system.")

# # import torch
# # print(torch.cuda.is_available())
# # print(torch.cuda.device_count())  # Should show 2
# # print(torch.cuda.get_device_name(1))  # Integrated GPU
# # tensor = torch.tensor([1.0, 2.0]).to('cuda:1')  # Use integrated GPU
# # print(tensor * 2)

import inspect, transformers
from transformers import TrainingArguments

print("🤖 transformers version:", transformers.__version__)
print("🗂 TrainingArguments loaded from module:", TrainingArguments.__module__)
print("📂 transformers package file:", transformers.__file__)
print("🔍 __init__ signature:", inspect.signature(TrainingArguments.__init__))
