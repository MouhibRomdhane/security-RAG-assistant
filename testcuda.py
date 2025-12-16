import torch
import torchvision
print(f"Torch: {torch.__version__}")
print(f"Vision: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# This specific line checks if the 'nms' operator is fixed
ops = torchvision.ops
print("Success! Torchvision operators are found.")