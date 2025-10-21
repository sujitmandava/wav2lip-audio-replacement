import torch
import time
import numpy as np
from Wav2Lip.models.wav2lip import Wav2Lip
from Wav2Lip.models.wav2lip_optimized import Wav2LipOptimized
from gpu_diagnostics import GPUDiagnostics

def benchmark_comparison(input_size=(8, 5, 1, 80, 16), face_size=(8, 6, 96, 96)):
    diag = GPUDiagnostics()
    device = diag.device
    
    # Print system info
    diag.print_system_info()
    
    # Create sample inputs
    audio_input = torch.randn(*input_size)
    face_input = torch.randn(*face_size)
    
    # Test original model
    print("\n=== Testing Original Model ===")
    model_original = Wav2Lip().to(device)
    
    def test_original():
        start_time = time.time()
        with torch.no_grad():
            output = model_original(audio_input.to(device), face_input.to(device))
        torch.cuda.synchronize()
        return time.time() - start_time
    
    # Test optimized model
    print("\n=== Testing Optimized Model ===")
    model_optimized = Wav2LipOptimized().to(device)
    
    def test_optimized():
        start_time = time.time()
        with torch.no_grad():
            output = model_optimized(audio_input.to(device), face_input.to(device))
        torch.cuda.synchronize()
        return time.time() - start_time
    
    # Run multiple iterations for each model
    n_iterations = 10
    times_original = []
    times_optimized = []
    
    print("\nRunning benchmark...")
    
    # Benchmark original model
    for i in range(n_iterations):
        torch.cuda.empty_cache()
        time_taken = test_original()
        times_original.append(time_taken)
        print(f"Original model - Iteration {i+1}: {time_taken:.4f}s")
    
    # Benchmark optimized model
    for i in range(n_iterations):
        torch.cuda.empty_cache()
        time_taken = test_optimized()
        times_optimized.append(time_taken)
        print(f"Optimized model - Iteration {i+1}: {time_taken:.4f}s")
    
    # Print results
    print("\n=== Benchmark Results ===")
    print(f"Original Model:")
    print(f"  Average time: {np.mean(times_original):.4f}s")
    print(f"  Std deviation: {np.std(times_original):.4f}s")
    print(f"\nOptimized Model:")
    print(f"  Average time: {np.mean(times_optimized):.4f}s")
    print(f"  Std deviation: {np.std(times_optimized):.4f}s")
    print(f"\nSpeedup: {np.mean(times_original)/np.mean(times_optimized):.2f}x")
    
    # Memory usage comparison
    print("\n=== Memory Usage Comparison ===")
    torch.cuda.empty_cache()
    
    # Original model memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model_original(audio_input.to(device), face_input.to(device))
    original_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    # Optimized model memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model_optimized(audio_input.to(device), face_input.to(device))
    optimized_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Original Model Peak Memory: {original_memory:.2f} MB")
    print(f"Optimized Model Peak Memory: {optimized_memory:.2f} MB")
    print(f"Memory Reduction: {(1 - optimized_memory/original_memory)*100:.1f}%")

if __name__ == "__main__":
    benchmark_comparison()