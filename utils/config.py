import os
from pathlib import Path

class Config:
    # GPU and Memory Settings
    BATCH_SIZE = 8
    MAX_MEMORY_USAGE = 0.8  # 80% GPU memory limit
    USE_MIXED_PRECISION = True
    
    # Directory Settings
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    TEMP_DIR = BASE_DIR / "temp"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    MODELS_DIR = BASE_DIR / "models"
    
    # Model Settings
    WAV2LIP_GAN_MODEL = MODELS_DIR / "Wav2Lip-SD-GAN.pt"
    WAV2LIP_NOGAN_MODEL = MODELS_DIR / "Wav2Lip-SD-NOGAN.pt"
    FACE_DETECTION_MODEL = MODELS_DIR / "face_detection/s3fd-619a316812.pth"
    
    # Processing Settings
    MAX_VIDEO_DIMENSION = 1280  # Max width/height for input videos
    FACE_DETECTION_BATCH = 8
    AUDIO_SAMPLE_RATE = 16000
    
    # File Settings
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
    ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aac'}
    
    # Cleanup Settings
    CLEANUP_INTERVAL = 3600  # 1 hour
    MAX_TEMP_FILE_AGE = 24 * 3600  # 24 hours
    
    # Error Retry Settings
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    
    @classmethod
    def init(cls):
        """Initialize configuration and create required directories"""
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUTS_DIR, exist_ok=True)
        
        # Validate model files exist
        required_files = [
            cls.WAV2LIP_GAN_MODEL,
            cls.WAV2LIP_NOGAN_MODEL,
            cls.FACE_DETECTION_MODEL
        ]
        for file in required_files:
            if not file.exists():
                raise FileNotFoundError(f"Required model file not found: {file}")
    
    @classmethod
    def validate_file_type(cls, file_path: str, file_type: str = 'video'):
        """Validate file extensions"""
        ext = Path(file_path).suffix.lower()
        if file_type == 'video':
            return ext in cls.ALLOWED_VIDEO_EXTENSIONS
        elif file_type == 'audio':
            return ext in cls.ALLOWED_AUDIO_EXTENSIONS
        return False