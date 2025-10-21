import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from typing import Optional, Tuple

from utils.config import Config

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate: int = Config.AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_audio(self, 
                  audio_path: Path,
                  normalize: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file
        
        :param audio_path: Path to audio file
        :param normalize: Whether to normalize audio
        :return: Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            
            # Normalize if requested
            if normalize:
                audio = librosa.util.normalize(audio)
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {str(e)}")
            raise
    
    def extract_mel_features(self,
                           audio: np.ndarray,
                           fps: int = 25,
                           mel_step_size: int = 16) -> torch.Tensor:
        """
        Extract mel spectrogram features from audio
        
        :param audio: Audio data
        :param fps: Frames per second of the video
        :param mel_step_size: Step size for mel spectrogram
        :return: Mel spectrogram features tensor
        """
        try:
            # Calculate hop length based on fps
            hop_length = int(self.sample_rate / fps)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=80,
                hop_length=hop_length,
                win_length=hop_length * 4,
                fmin=90,
                fmax=7600
            )
            
            # Convert to log scale
            mel_spec = np.log(mel_spec + 1e-6)
            
            # Normalize
            mel_mean = mel_spec.mean()
            mel_std = mel_spec.std()
            mel_spec = (mel_spec - mel_mean) / mel_std
            
            # Convert to tensor
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
            
            # Split into windows
            mel_windows = []
            for i in range(0, mel_tensor.size(2) - mel_step_size + 1, mel_step_size):
                mel_windows.append(mel_tensor[:, :, i:i+mel_step_size])
            
            return torch.stack(mel_windows).to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to extract mel features: {str(e)}")
            raise
    
    def save_audio(self,
                  audio: np.ndarray,
                  output_path: Path,
                  sample_rate: Optional[int] = None):
        """
        Save audio to file
        
        :param audio: Audio data
        :param output_path: Output file path
        :param sample_rate: Sample rate (defaults to self.sample_rate)
        """
        try:
            sr = sample_rate or self.sample_rate
            sf.write(str(output_path), audio, sr)
            
        except Exception as e:
            logger.error(f"Failed to save audio file {output_path}: {str(e)}")
            raise
    
    @staticmethod
    def get_audio_duration(audio_path: Path) -> float:
        """Get duration of audio file in seconds"""
        try:
            return librosa.get_duration(filename=str(audio_path))
        except Exception as e:
            logger.error(f"Failed to get audio duration for {audio_path}: {str(e)}")
            raise