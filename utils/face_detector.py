import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

from utils.config import Config
from utils.memory_manager import memory_manager
from utils.model_loader import model_loader

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.batch_size = Config.FACE_DETECTION_BATCH
        
    def load_model(self):
        """Load the face detection model"""
        if self.model is None:
            self.model = model_loader.load_model(
                Config.FACE_DETECTION_MODEL,
                'face_detection'
            )
    
    def detect_faces_batch(self, 
                          frames: List[np.ndarray],
                          confidence_threshold: float = 0.98) -> List[Optional[np.ndarray]]:
        """
        Detect faces in a batch of frames
        
        :param frames: List of frames to process
        :param confidence_threshold: Minimum confidence score for face detection
        :return: List of detected face coordinates (or None if no face detected)
        """
        self.load_model()
        
        # Process in batches
        batch_size = memory_manager.optimize_gpu_allocation(self.batch_size)
        results = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Prepare batch
            batch_tensor = torch.stack([
                self._preprocess_frame(frame) for frame in batch_frames
            ]).to(self.device)
            
            # Detect faces
            with torch.no_grad():
                try:
                    if Config.USE_MIXED_PRECISION:
                        with torch.cuda.amp.autocast():
                            detections = self.model(batch_tensor)
                    else:
                        detections = self.model(batch_tensor)
                    
                    # Process detections
                    batch_results = self._process_detections(
                        detections, batch_frames, confidence_threshold
                    )
                    results.extend(batch_results)
                    
                except Exception as e:
                    logger.error(f"Face detection failed: {str(e)}")
                    results.extend([None] * len(batch_frames))
        
        return results
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for face detection"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize
        frame_float = frame_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(frame_float).permute(2, 0, 1)
        
        return tensor
    
    def _process_detections(self,
                          detections: torch.Tensor,
                          original_frames: List[np.ndarray],
                          confidence_threshold: float) -> List[Optional[np.ndarray]]:
        """Process face detection results"""
        results = []
        
        for i, frame in enumerate(original_frames):
            frame_dets = detections[i]
            
            # Filter by confidence
            confident_dets = frame_dets[frame_dets[:, -1] > confidence_threshold]
            
            if len(confident_dets) == 0:
                results.append(None)
                continue
            
            # Get the largest face
            areas = (confident_dets[:, 2] - confident_dets[:, 0]) * \
                   (confident_dets[:, 3] - confident_dets[:, 1])
            largest_face = confident_dets[areas.argmax()]
            
            # Convert to pixel coordinates
            h, w = frame.shape[:2]
            bbox = (largest_face[:4] * torch.tensor([w, h, w, h])
                   .to(largest_face.device)).cpu().numpy()
            
            results.append(bbox)
        
        return results
    
    def extract_face_region(self,
                          frame: np.ndarray,
                          bbox: np.ndarray,
                          scale: float = 1.1) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract and preprocess face region from frame
        
        :param frame: Input frame
        :param bbox: Bounding box coordinates [x1, y1, x2, y2]
        :param scale: Scale factor for face region
        :return: Tuple of (face_region, actual_bbox)
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate center and size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        size = max(x2 - x1, y2 - y1) * scale
        
        # Calculate new bbox
        new_x1 = int(max(center_x - size/2, 0))
        new_y1 = int(max(center_y - size/2, 0))
        new_x2 = int(min(center_x + size/2, frame.shape[1]))
        new_y2 = int(min(center_y + size/2, frame.shape[0]))
        
        # Extract region
        face_region = frame[new_y1:new_y2, new_x1:new_x2]
        
        return face_region, (new_x1, new_y1, new_x2, new_y2)
    
    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            model_loader.unload_model(Config.FACE_DETECTION_MODEL)
            self.model = None