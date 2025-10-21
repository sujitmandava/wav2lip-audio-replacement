import torch
import sys

def check_gpu_config():
    print("=== Python Version ===")
    print(f"Python version: {sys.version}")
    
    print("\n=== PyTorch Version ===")
    print(f"PyTorch version: {torch.__version__}")
    
    print("\n=== CUDA Configuration ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA memory operations
        print("\n=== Testing CUDA Memory Operations ===")
        try:
            x = torch.rand(1000, 1000).cuda()
            print(f"Successfully allocated {x.element_size() * x.nelement() / 1024 / 1024:.2f} MB on GPU")
            del x
            torch.cuda.empty_cache()
            print("Memory test successful")
        except Exception as e:
            print(f"Error during memory test: {str(e)}")
    
    print("\n=== GPU Properties ===")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {props.name}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1024**2:.0f} MB")
            print(f"  Multi-processor count: {props.multi_processor_count}")
    else:
        print("No GPU devices available")

if __name__ == "__main__":
    check_gpu_config()