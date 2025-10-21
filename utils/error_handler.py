import logging
import traceback
from typing import Optional, Any, Dict
from pathlib import Path
import json
import time
from functools import wraps

logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Base exception for processing errors"""
    def __init__(self, message: str, error_code: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class FileError(ProcessingError):
    """File-related errors"""
    pass

class ModelError(ProcessingError):
    """Model-related errors"""
    pass

class GPUError(ProcessingError):
    """GPU-related errors"""
    pass

def log_error(error: Exception,
              context: Optional[Dict[str, Any]] = None,
              log_path: Optional[Path] = None):
    """
    Log error with context
    
    :param error: The exception to log
    :param context: Additional context information
    :param log_path: Optional path to write error log
    """
    error_info = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'context': context or {}
    }
    
    # Log to console
    logger.error(f"Error occurred: {error_info['error_type']}: {error_info['error_message']}")
    if context:
        logger.error(f"Context: {json.dumps(context, indent=2)}")
    
    # Write to file if specified
    if log_path:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'a') as f:
                json.dump(error_info, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write error log: {str(e)}")

def retry_on_error(max_retries: int = 3,
                  delay: float = 1.0,
                  exceptions: tuple = (Exception,)):
    """
    Decorator to retry function on error
    
    :param max_retries: Maximum number of retry attempts
    :param delay: Delay between retries in seconds
    :param exceptions: Tuple of exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed. "
                            f"Last error: {str(e)}"
                        )
            
            raise last_error
            
        return wrapper
    return decorator

def validate_file_exists(file_path: Path):
    """
    Validate file exists and is accessible
    
    :param file_path: Path to file
    :raises FileError: If file doesn't exist or is inaccessible
    """
    if not file_path.exists():
        raise FileError(
            f"File not found: {file_path}",
            'FILE_NOT_FOUND'
        )
    
    if not file_path.is_file():
        raise FileError(
            f"Not a file: {file_path}",
            'INVALID_FILE'
        )
    
    try:
        with open(file_path, 'rb'):
            pass
    except Exception as e:
        raise FileError(
            f"File not accessible: {file_path}. Error: {str(e)}",
            'FILE_ACCESS_ERROR'
        )

def check_gpu_availability():
    """
    Check if GPU is available and working
    
    :raises GPUError: If GPU is not available or not working properly
    """
    import torch
    
    if not torch.cuda.is_available():
        raise GPUError(
            "GPU not available",
            'GPU_NOT_AVAILABLE'
        )
    
    try:
        # Try to allocate a small tensor on GPU
        torch.cuda.empty_cache()
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
    except Exception as e:
        raise GPUError(
            f"GPU test failed: {str(e)}",
            'GPU_TEST_FAILED'
        )

def format_error_response(error: Exception) -> Dict[str, Any]:
    """Format error for API response"""
    if isinstance(error, ProcessingError):
        response = {
            'error': True,
            'error_code': error.error_code,
            'message': str(error),
            'details': error.details
        }
    else:
        response = {
            'error': True,
            'error_code': 'UNKNOWN_ERROR',
            'message': str(error)
        }
    
    return response