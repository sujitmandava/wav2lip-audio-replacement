import torch
import time
import psutil
import os
from Wav2Lip.models.wav2lip import Wav2Lip

class GPUDiagnostics:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def print_system_info(self):
        print("\n=== System Information ===")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current Device: {torch.cuda.current_device()}")
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Device Capability: {torch.cuda.get_device_capability()}")
            print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")

    def monitor_gpu_memory(self):
        if not torch.cuda.is_available():
            print("GPU not available")
            return
            
        print("\n=== GPU Memory Usage ===")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    def check_model_device_placement(self, model):
        print("\n=== Model Device Placement ===")
        on_gpu = 0
        total = 0
        for name, param in model.named_parameters():
            total += 1
            if param.is_cuda:
                on_gpu += 1
            else:
                print(f"Warning: {name} is not on GPU")
        
        print(f"Parameters on GPU: {on_gpu}/{total} ({on_gpu/total*100:.1f}%)")

    def monitor_operation(self, operation_name, func):
        print(f"\n=== Monitoring {operation_name} ===")
        
        # Record initial state
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        start_time = time.time()
        
        # Run operation
        result = func()
        
        # Record final state
        end_time = time.time()
        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Memory change: {(end_mem - start_mem) / 1024**2:.2f} MB")
        
        return result

def run_diagnostics():
    diag = GPUDiagnostics()
    
    # Print system information
    diag.print_system_info()
    
    # Initial memory state
    diag.monitor_gpu_memory()
    
    # Load and check model
    def load_model():
        model = Wav2Lip()
        model = model.to(diag.device)
        return model
    
    model = diag.monitor_operation("Model Loading", load_model)
    
    # Check model device placement
    diag.check_model_device_placement(model)
    
    # Final memory state
    diag.monitor_gpu_memory()

if __name__ == "__main__":
    run_diagnostics()