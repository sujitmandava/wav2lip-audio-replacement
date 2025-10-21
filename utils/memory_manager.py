import torch
import gc
import logging
from typing import Optional
import psutil
import os

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, max_memory_usage: float = 0.8):
        """
        Initialize memory manager
        :param max_memory_usage: Maximum fraction of GPU memory to use (0.0 to 1.0)
        """
        self.max_memory_usage = max_memory_usage
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def check_gpu_memory(self) -> tuple[float, float]:
        """
        Check current GPU memory usage
        :return: Tuple of (used_memory_fraction, free_memory_gb)
        """
        if not torch.cuda.is_available():
            return 0.0, 0.0
            
        total_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated()
        free_memory = total_memory - used_memory
        
        used_fraction = used_memory / total_memory
        free_memory_gb = free_memory / (1024**3)  # Convert to GB
        
        return used_fraction, free_memory_gb
    
    def check_cpu_memory(self) -> tuple[float, float]:
        """
        Check current CPU memory usage
        :return: Tuple of (used_memory_fraction, free_memory_gb)
        """
        vm = psutil.virtual_memory()
        used_fraction = vm.percent / 100.0
        free_memory_gb = vm.available / (1024**3)
        
        return used_fraction, free_memory_gb
    
    def cleanup(self, force: bool = False):
        """
        Clean up unused memory
        :param force: If True, perform aggressive cleanup
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if force:
            gc.collect()
            
        used_gpu, free_gpu = self.check_gpu_memory()
        used_cpu, free_cpu = self.check_cpu_memory()
        
        logger.info(f"Memory after cleanup - GPU: {used_gpu:.1%} used ({free_gpu:.1f}GB free), "
                   f"CPU: {used_cpu:.1%} used ({free_cpu:.1f}GB free)")
    
    def optimize_gpu_allocation(self, batch_size: Optional[int] = None) -> int:
        """
        Optimize batch size based on available GPU memory
        :param batch_size: Requested batch size (None for auto-detection)
        :return: Optimized batch size
        """
        if not torch.cuda.is_available():
            return 1
            
        used_fraction, _ = self.check_gpu_memory()
        available_fraction = self.max_memory_usage - used_fraction
        
        if available_fraction <= 0.1:  # Less than 10% GPU memory available
            self.cleanup(force=True)
            used_fraction, _ = self.check_gpu_memory()
            available_fraction = self.max_memory_usage - used_fraction
        
        if batch_size is None:
            # Auto-detect based on available memory
            if available_fraction > 0.5:
                return 8
            elif available_fraction > 0.3:
                return 4
            elif available_fraction > 0.2:
                return 2
            return 1
        
        # Adjust requested batch size if needed
        if used_fraction + (available_fraction / 2) > self.max_memory_usage:
            adjusted_batch_size = max(1, batch_size // 2)
            logger.warning(f"Reducing batch size from {batch_size} to {adjusted_batch_size} due to memory constraints")
            return adjusted_batch_size
            
        return batch_size
    
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor memory during an operation"""
        class MemoryMonitor:
            def __init__(self, manager, op_name):
                self.manager = manager
                self.op_name = op_name
                self.start_gpu = 0
                self.start_cpu = 0
            
            def __enter__(self):
                if torch.cuda.is_available():
                    self.start_gpu = torch.cuda.memory_allocated()
                self.start_cpu = psutil.Process(os.getpid()).memory_info().rss
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if torch.cuda.is_available():
                    end_gpu = torch.cuda.memory_allocated()
                    gpu_diff = (end_gpu - self.start_gpu) / (1024**2)  # MB
                else:
                    gpu_diff = 0
                    
                end_cpu = psutil.Process(os.getpid()).memory_info().rss
                cpu_diff = (end_cpu - self.start_cpu) / (1024**2)  # MB
                
                logger.info(f"Memory change during {self.op_name}: "
                           f"GPU: {gpu_diff:+.1f}MB, CPU: {cpu_diff:+.1f}MB")
                
                if exc_type is not None:
                    self.manager.cleanup(force=True)
        
        return MemoryMonitor(self, operation_name)

# Global memory manager instance
memory_manager = MemoryManager()