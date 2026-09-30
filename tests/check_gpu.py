import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    try:
        x = torch.randn(10, 10).cuda() * 2
        print(f"Tensor Op Result: {x.sum().item():.4f}")
        print("GPU Hardware Gate PASSED successfully!")
    except Exception as e:
        print(f"GPU Hardware Gate FAILED: {e}")
else:
    print("CUDA is NOT available in this PyTorch build.")
