import os
import logging
from pathlib import Path
import torch
import subprocess
from typing import Optional

from utils.config import Config
from utils.error_handler import ProcessingError

logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the processing environment"""
    # Create required directories
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUTS_DIR, exist_ok=True)
    
    # Set up logging
    setup_logging()
    
    # Check CUDA availability
    check_cuda()
    
    # Verify model files
    verify_models()
    
    # Set up environment variables
    setup_env_vars()

def setup_logging():
    """Configure logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler()
        ]
    )

def check_cuda():
    """Check CUDA availability and setup"""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Running in CPU mode")
        return
    
    # Log CUDA information
    cuda_version = torch.version.cuda
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    
    logger.info(f"CUDA version: {cuda_version}")
    logger.info(f"GPU device: {device_name}")
    logger.info(f"Number of GPU devices: {device_count}")
    
    # Test CUDA memory allocation
    try:
        torch.cuda.empty_cache()
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
    except Exception as e:
        logger.error(f"CUDA initialization failed: {str(e)}")
        raise ProcessingError("CUDA initialization failed", "CUDA_INIT_ERROR")

def verify_models():
    """Verify required model files exist"""
    required_models = [
        Config.WAV2LIP_GAN_MODEL,
        Config.WAV2LIP_NOGAN_MODEL,
        Config.FACE_DETECTION_MODEL
    ]
    
    missing_models = []
    for model_path in required_models:
        if not model_path.exists():
            missing_models.append(model_path)
    
    if missing_models:
        error_msg = f"Missing required model files: {', '.join(map(str, missing_models))}"
        logger.error(error_msg)
        raise ProcessingError(error_msg, "MISSING_MODELS")

def setup_env_vars():
    """Set up environment variables"""
    # Set number of threads for OpenMP
    os.environ["OMP_NUM_THREADS"] = "4"
    
    # Set CUDA device order
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Disable OpenCV DNN CUDA backend (can be unstable)
    os.environ["OPENCV_DNN_BACKEND_CUDA"] = "0"

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    except FileNotFoundError:
        logger.error("ffmpeg not found. Required for audio processing")
        raise ProcessingError("ffmpeg not found", "FFMPEG_NOT_FOUND")

def cleanup_temp_files(max_age: Optional[int] = None):
    """Clean up temporary files"""
    max_age = max_age or Config.MAX_TEMP_FILE_AGE
    
    try:
        for file in Config.TEMP_DIR.glob("*"):
            if file.is_file():
                file_age = time.time() - file.stat().st_mtime
                if file_age > max_age:
                    file.unlink()
    except Exception as e:
        logger.warning(f"Failed to cleanup temp files: {str(e)}")

if __name__ == "__main__":
    setup_environment()
    check_ffmpeg()
    logger.info("Environment setup complete")