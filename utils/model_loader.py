import torch
import logging
from pathlib import Path
from typing import Optional, Union
import time

from utils.config import Config
from utils.memory_manager import memory_manager

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.last_used = {}
    
    def load_model(self, 
                  model_path: Union[str, Path], 
                  model_type: str,
                  force_reload: bool = False) -> Optional[torch.nn.Module]:
        """
        Load a model with proper error handling and GPU optimization
        
        :param model_path: Path to the model file
        :param model_type: Type of model ('wav2lip' or 'face_detection')
        :param force_reload: Force model reloading even if already loaded
        :return: Loaded model or None if loading fails
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Check if model is already loaded
        if not force_reload and model_path in self.models:
            self.last_used[model_path] = time.time()
            return self.models[model_path]
            
        try:
            with memory_manager.monitor_operation(f"Loading {model_type} model"):
                if model_type == 'wav2lip':
                    from Wav2Lip.models.wav2lip import Wav2Lip
                    model = Wav2Lip()
                elif model_type == 'face_detection':
                    from Wav2Lip.face_detection.detection.sfd.net_s3fd import s3fd
                    model = s3fd()
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Load state dict
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                
                # Move to GPU if available
                model = model.to(self.device)
                
                # Set to eval mode
                model.eval()
                
                # Enable mixed precision if available
                if Config.USE_MIXED_PRECISION and torch.cuda.is_available():
                    model = model.half()
                
                self.models[model_path] = model
                self.last_used[model_path] = time.time()
                
                logger.info(f"Successfully loaded {model_type} model from {model_path}")
                return model
                
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise
    
    def unload_model(self, model_path: Union[str, Path]):
        """Unload a model and free its memory"""
        model_path = Path(model_path)
        if model_path in self.models:
            del self.models[model_path]
            del self.last_used[model_path]
            torch.cuda.empty_cache()
    
    def cleanup_unused_models(self, max_age: int = 3600):
        """Unload models that haven't been used for a while"""
        current_time = time.time()
        for path, last_used in list(self.last_used.items()):
            if current_time - last_used > max_age:
                logger.info(f"Unloading unused model: {path}")
                self.unload_model(path)
    
    def get_model_memory_usage(self, model_path: Union[str, Path]) -> float:
        """Get the memory usage of a loaded model in MB"""
        model_path = Path(model_path)
        if model_path not in self.models:
            return 0.0
            
        model = self.models[model_path]
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

# Global model loader instance
model_loader = ModelLoader()