import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    try:
        t = torch.tensor([1, 2], device="cuda")
        print("Tensor on CUDA created successfully.")
    except Exception as e:
        print(f"Tensor creation failed: {e}")
else:
    print("CUDA not available.")
