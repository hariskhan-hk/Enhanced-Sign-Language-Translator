import torch

def check_cuda():
    # Basic CUDA availability
    print("CUDA Available:", torch.cuda.is_available())
    
    # GPU Details
    if torch.cuda.is_available():
        print("GPU Device Name:", torch.cuda.get_device_name(0))
        gpu_props = torch.cuda.get_device_properties(0)
        
        print("\nGPU Specifications:")
        print(f"Total Memory: {gpu_props.total_memory / 1e9:.2f} GB")
        print(f"CUDA Capability: {gpu_props.major}.{gpu_props.minor}")
        print(f"Multi-Processor Count: {gpu_props.multi_processor_count}")
    
    # PyTorch Version
    print("\nPyTorch Version:", torch.__version__)
    print("CUDA Version:", torch.version.cuda)

if __name__ == "__main__":
    check_cuda()
