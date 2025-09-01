#!/usr/bin/env python3
"""
FastAPI Lip Sync Service - Fixed version with natural face preservation
Key fixes:
1. Minimal mouth-region blending only
2. Better face coordinate handling
3. Preserved original face features
4. Improved lighting/color matching
"""

import os
import sys
import cv2
import numpy as np
import torch
import librosa
import subprocess
import soundfile as sf
from tqdm import tqdm
import tempfile
import shutil
import uuid
from datetime import datetime
from pathlib import Path
import asyncio
import aiofiles
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

import logging
import traceback

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Response models
class ProcessingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_time: Optional[str] = None

class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    message: str
    output_url: Optional[str] = None
    error: Optional[str] = None

# Global job storage
jobs = {}

class LipSyncPipeline:
    def __init__(self, use_gan=True):
        """Initialize the lip sync pipeline in CPU-only mode"""
        self.device = 'cpu'
        self.use_gan = use_gan

        # Add Wav2Lip repo to path
        sys.path.append('Wav2Lip')

        # Select Wav2Lip model path
        self.model_path = "models/Wav2Lip-SD-GAN.pt" if use_gan else "models/Wav2Lip-SD-NOGAN.pt"
        self.s3fd_model_path = "models/face_detection/s3fd-619a316812.pth"

        self.model = None
        self.face_detect = None
        self._model_loaded = False

    def load_models(self):
        """Load Wav2Lip and face detection models"""
        if self._model_loaded:
            return
            
        try:
            logger.info(f"Loading {'GAN' if self.use_gan else 'Non-GAN'} model on CPU...")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Wav2Lip model not found: {self.model_path}")

            from Wav2Lip.models import Wav2Lip
            from Wav2Lip.face_detection.detection.sfd.net_s3fd import s3fd

            try:
                logger.info("Attempting to load as PyTorch state dict...")
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                
                model = Wav2Lip()
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                self.model = model
                logger.info("Successfully loaded PyTorch model")
                
            except Exception as e:
                logger.warning(f"PyTorch loading failed: {e}, trying TorchScript...")
                try:
                    self.model = torch.jit.load(self.model_path, map_location='cpu')
                    self.model.eval()
                    logger.info("Successfully loaded TorchScript model")
                except Exception as e2:
                    raise Exception(f"Both loading methods failed. PyTorch: {e}, TorchScript: {e2}")

            # Load face detector
            if not os.path.exists(self.s3fd_model_path):
                raise FileNotFoundError(f"S3FD model not found: {self.s3fd_model_path}")

            self.face_detect = s3fd()
            self.face_detect.load_state_dict(torch.load(self.s3fd_model_path, map_location='cpu', weights_only=False))
            self.face_detect.eval()

            self._model_loaded = True
            logger.info("All models loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(traceback.format_exc())
            raise

    def extract_frames_and_audio(self, video_path):
        """Extract frames with better strategy - process every 3rd frame instead of 1 per second"""
        try:
            logger.info(f"Extracting frames from: {video_path}")
            temp_dir = tempfile.mkdtemp()
            frames_dir = os.path.join(temp_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25

            all_frames = []
            keyframes = []
            keyframe_indices = []
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            # Process every 3rd frame instead of 1 per second for smoother results
            frame_skip = 3

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                all_frames.append(frame)
                
                # Process every 3rd frame for better temporal resolution
                if frame_count % frame_skip == 0:
                    keyframes.append(frame.copy())
                    keyframe_indices.append(frame_count)

                frame_count += 1

            cap.release()
            logger.info(f"Extracted {len(all_frames)} total frames, {len(keyframes)} keyframes from {duration:.2f}s video at {fps:.2f} FPS")
            
            return {
                'all_frames': all_frames,
                'keyframes': keyframes, 
                'keyframe_indices': keyframe_indices,
                'fps': fps,
                'temp_dir': temp_dir,
                'duration': duration,
                'frame_skip': frame_skip
            }
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            logger.error(traceback.format_exc())
            raise

    def preprocess_audio(self, audio_path, target_duration):
        """Preprocess audio to match video duration"""
        try:
            logger.info(f"Preprocessing audio: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000)
            target_length = int(target_duration * sr)

            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                repeats = int(np.ceil(target_length / len(audio)))
                audio = np.tile(audio, repeats)[:target_length]

            logger.info(f"Processed audio length: {len(audio)} samples at {sr} Hz")
            return audio, sr
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            logger.error(traceback.format_exc())
            raise

    def get_mel_chunks(self, audio, sr, fps):
        """Produce mel chunks shaped (80, 16) that match Wav2Lip expectations"""
        try:
            n_mels = 80
            n_fft = 800
            hop_length = 200
            win_length = 800
            fmin = 55
            fmax = 7600

            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax
            )

            mel_spec = np.log(np.maximum(1e-9, mel_spec))
            mel_step_size = 16
            chunks = []
            n_frames = mel_spec.shape[1]

            if n_frames < mel_step_size:
                pad_amt = mel_step_size - n_frames
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_amt)), mode='edge')
                n_frames = mel_spec.shape[1]

            for i in range(0, n_frames - mel_step_size + 1, mel_step_size):
                chunk = mel_spec[:, i:i + mel_step_size]
                chunks.append(chunk)

            if len(chunks) == 0:
                chunks.append(mel_spec[:, -mel_step_size:])

            logger.info(f"Mel spectrogram shape: {mel_spec.shape}, created {len(chunks)} mel chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error in get_mel_chunks: {e}")
            logger.error(traceback.format_exc())
            raise

    def prepare_frame_input(self, frame, face_bbox):
        """Prepare frame input with proper normalization"""
        y1, y2, x1, x2 = face_bbox

        # Extract and resize face
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (96, 96))

        # Convert to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5

        # Create masked face (set mouth region to 0)
        masked_face = face.copy()
        masked_face[48:96, :] = 0

        # Stack original + masked face → 6 channels
        stacked_face = np.concatenate([face, masked_face], axis=2)

        # Convert to tensor: [H,W,C] → [1,C,H,W]
        face_tensor = torch.FloatTensor(stacked_face).permute(2, 0, 1).unsqueeze(0)

        return face_tensor

    def create_precise_lip_mask(self, face_height, face_width):
        """Create a precise mask for only the lip/mouth area"""
        mask = np.zeros((face_height, face_width), dtype=np.float32)
        
        # Much smaller mouth region - only the actual lips
        mouth_center_y = int(face_height * 0.75)  # 75% down from top
        mouth_center_x = int(face_width * 0.5)    # Center horizontally
        
        # Smaller mouth dimensions
        mouth_width = int(face_width * 0.25)      # Only 25% of face width
        mouth_height = int(face_height * 0.12)    # Only 12% of face height
        
        # Create elliptical mask for natural lip shape
        for y in range(face_height):
            for x in range(face_width):
                # Distance from mouth center
                dx = (x - mouth_center_x) / (mouth_width / 2)
                dy = (y - mouth_center_y) / (mouth_height / 2)
                
                # Elliptical distance
                ellipse_dist = np.sqrt(dx*dx + dy*dy)
                
                if ellipse_dist <= 1.0:
                    # Soft falloff for natural blending
                    mask[y, x] = max(0, 0.7 * (1.0 - ellipse_dist**2))
        
        # Apply strong Gaussian blur for very smooth transitions
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask

    def blend_face_natural(self, original_frame, predicted_face, face_bbox):
        """Natural face blending that preserves all face features except lips"""
        y1, y2, x1, x2 = face_bbox
        face_height = y2 - y1
        face_width = x2 - x1
        
        # Resize predicted face to match bbox
        predicted_face_resized = cv2.resize(predicted_face, (face_width, face_height))
        
        # Get original face region
        original_face = original_frame[y1:y2, x1:x2].copy()
        
        # Create precise lip-only mask
        lip_mask = self.create_precise_lip_mask(face_height, face_width)
        lip_mask = np.expand_dims(lip_mask, axis=2)
        
        # Color correction - match predicted face to original lighting
        predicted_corrected = self.match_face_color(original_face, predicted_face_resized)
        
        # Apply minimal blending only to lip region
        blended_face = (
            lip_mask * predicted_corrected + 
            (1 - lip_mask) * original_face
        ).astype(np.uint8)
        
        # Copy back to original frame
        result = original_frame.copy()
        result[y1:y2, x1:x2] = blended_face
        
        return result

    def match_face_color(self, original_face, predicted_face):
        """Match color and lighting of predicted face to original"""
        original_face = original_face.astype(np.float32)
        predicted_face = predicted_face.astype(np.float32)
        
        # Match histogram for each channel
        corrected = predicted_face.copy()
        
        for channel in range(3):
            orig_channel = original_face[:, :, channel]
            pred_channel = predicted_face[:, :, channel]
            
            # Calculate statistics
            orig_mean = np.mean(orig_channel)
            orig_std = np.std(orig_channel)
            pred_mean = np.mean(pred_channel)
            pred_std = np.std(pred_channel)
            
            # Normalize and match
            if pred_std > 0:
                corrected[:, :, channel] = (
                    (pred_channel - pred_mean) * (orig_std / pred_std) + orig_mean
                )
        
        # Ensure values are in valid range
        corrected = np.clip(corrected, 0, 255)
        return corrected.astype(np.uint8)

    def process_keyframes_with_progress(self, frame_data, audio, sr, job_id):
        """Process keyframes with better face preservation"""
        try:
            keyframes = frame_data['keyframes']
            mel_chunks = self.get_mel_chunks(audio, sr, frame_data['fps'])
            
            if mel_chunks is None or len(mel_chunks) == 0:
                raise ValueError("Failed to generate mel-spectrogram chunks")
            
            processed_keyframes = []
            total_keyframes = len(keyframes)
            
            logger.info(f"Processing {total_keyframes} keyframes with {len(mel_chunks)} mel chunks")
            
            # Initialize face detector
            from Wav2Lip.face_detection.detection.sfd.sfd_detector import SFDDetector
            detect_face = SFDDetector(device='cpu', path_to_detector="models/face_detection/s3fd-619a316812.pth")
            
            for i in range(total_keyframes):
                # Update progress
                jobs[job_id]['progress'] = (i + 1) / total_keyframes * 70  # Reserve 30% for video creation
                jobs[job_id]['message'] = f"Processing keyframe {i+1}/{total_keyframes}"

                frame = keyframes[i]
                
                # Better mel chunk alignment
                time_ratio = i / max(1, len(keyframes) - 1)
                mel_chunk_idx = min(int(time_ratio * len(mel_chunks)), len(mel_chunks) - 1)
                mel_chunk = mel_chunks[mel_chunk_idx]

                # Face detection with error handling
                try:
                    faces = detect_face.detect_from_image(frame)
                    if len(faces) == 0:
                        logger.warning(f"No face detected in keyframe {i}")
                        processed_keyframes.append(frame)
                        continue
                except Exception as e:
                    logger.error(f"Face detection failed for keyframe {i}: {e}")
                    processed_keyframes.append(frame)
                    continue

                # Get largest face (most prominent)
                face_bbox = max(faces, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
                x1, y1, x2, y2 = map(int, face_bbox[:4])
                
                # Validate and constrain coordinates
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid face bbox in keyframe {i}")
                    processed_keyframes.append(frame)
                    continue

                # Model inference
                try:
                    frame_tensor = self.prepare_frame_input(frame, (y1, y2, x1, x2))
                    mel_tensor = torch.FloatTensor(mel_chunk).unsqueeze(0).unsqueeze(0)

                    with torch.no_grad():
                        pred = self.model(mel_tensor, frame_tensor)
                        
                        # Convert prediction back to image
                        pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        pred = np.clip((pred * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
                        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
                        
                        # Use natural blending to preserve face features
                        result_frame = self.blend_face_natural(frame, pred, (y1, y2, x1, x2))
                        processed_keyframes.append(result_frame)
                        
                except Exception as e:
                    logger.error(f"Model inference failed for keyframe {i}: {e}")
                    processed_keyframes.append(frame)

            return processed_keyframes
            
        except Exception as e:
            logger.error(f"Error in process_keyframes_with_progress: {e}")
            logger.error(traceback.format_exc())
            raise

    def create_full_video_from_keyframes(self, frame_data, processed_keyframes, job_id):
        """Create full video with smooth interpolation between processed keyframes"""
        try:
            all_frames = frame_data['all_frames']
            keyframe_indices = frame_data['keyframe_indices']
            fps = frame_data['fps']
            frame_skip = frame_data['frame_skip']
            
            total_frames = len(all_frames)
            full_video_frames = []
            
            logger.info(f"Creating full video from {len(processed_keyframes)} keyframes for {total_frames} total frames")
            
            for frame_idx in range(total_frames):
                # Update progress
                if frame_idx % 50 == 0:
                    progress = 70 + (frame_idx / total_frames) * 25  # 70-95%
                    jobs[job_id]['progress'] = progress
                    jobs[job_id]['message'] = f"Creating full video: frame {frame_idx+1}/{total_frames}"
                
                original_frame = all_frames[frame_idx]
                
                # Find closest processed keyframes for interpolation
                keyframe_idx = frame_idx // frame_skip
                keyframe_idx = min(keyframe_idx, len(processed_keyframes) - 1)
                
                processed_frame = processed_keyframes[keyframe_idx]
                
                # Use less blending for more natural results
                position_in_group = (frame_idx % frame_skip) / frame_skip
                
                # Lighter blending - preserve more of original
                blend_weight = 0.15 + 0.1 * np.sin(position_in_group * np.pi)  # 0.15 to 0.25
                
                # Gentle blending
                blended_frame = cv2.addWeighted(
                    original_frame, 1.0 - blend_weight,
                    processed_frame, blend_weight,
                    0
                )
                
                full_video_frames.append(blended_frame)
            
            logger.info(f"Created full video with {len(full_video_frames)} frames")
            return full_video_frames
            
        except Exception as e:
            logger.error(f"Error creating full video: {e}")
            logger.error(traceback.format_exc())
            raise

    def create_output_video(self, frames, audio, sr, fps, output_path):
        """Save video using OpenCV and merge audio using FFmpeg"""
        try:
            # Step 1: Save frames as video
            temp_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            for frame in frames:
                out.write(frame)
            out.release()

            # Step 2: Save audio
            temp_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(temp_audio_path, audio, sr)

            # Step 3: Merge with FFmpeg
            try:
                ffmpeg_path = r"D:\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
                cmd = [
                    ffmpeg_path,
                    "-y",
                    "-i", temp_video_path,
                    "-i", temp_audio_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",  # Higher quality
                    "-c:a", "aac",
                    "-b:a", "192k",  # Better audio quality
                    "-shortest",
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info("FFmpeg merge successful")

            except Exception as e:
                logger.error(f"FFmpeg merge failed: {e}")
                shutil.move(temp_video_path, output_path)

            # Cleanup
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
        except Exception as e:
            logger.error(f"Error creating output video: {e}")
            logger.error(traceback.format_exc())
            raise

    async def process_async(self, input_video_path, new_audio_path, output_path, job_id):
        """Async processing pipeline with natural face preservation"""
        try:
            logger.info(f"Starting job {job_id} with natural face preservation")
            jobs[job_id]['status'] = 'processing'
            jobs[job_id]['message'] = 'Loading models...'

            self.load_models()

            jobs[job_id]['message'] = 'Extracting frames...'
            frame_data = self.extract_frames_and_audio(input_video_path)

            jobs[job_id]['message'] = 'Processing audio...'
            audio, sr = self.preprocess_audio(new_audio_path, frame_data['duration'])

            jobs[job_id]['message'] = 'Starting natural lip sync processing...'
            processed_keyframes = self.process_keyframes_with_progress(frame_data, audio, sr, job_id)

            jobs[job_id]['message'] = 'Creating natural full-speed video...'
            full_video_frames = self.create_full_video_from_keyframes(frame_data, processed_keyframes, job_id)

            jobs[job_id]['message'] = 'Encoding final video...'
            jobs[job_id]['progress'] = 95
            self.create_output_video(full_video_frames, audio, sr, frame_data['fps'], output_path)

            shutil.rmtree(frame_data['temp_dir'])

            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['message'] = 'Processing completed with natural face preservation!'
            jobs[job_id]['output_url'] = f"/download/{job_id}"

            logger.info(f"Job {job_id} completed successfully with natural results!")
        except Exception as e:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['message'] = f'Processing failed: {str(e)}'
            logger.error(f"Job {job_id} failed: {e}")
            logger.error(traceback.format_exc())

# Initialize FastAPI app
app = FastAPI(
    title="Natural Lip Sync API",
    description="Natural Audio Replacement with Face-Preserving Lip Sync",
    version="2.0.0"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = LipSyncPipeline(use_gan=True)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

@app.get("/")
async def serve_html():
    return FileResponse("static/index.html")

@app.post("/process", response_model=ProcessingResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    input_video: UploadFile = File(..., description="Input video file"),
    new_audio: UploadFile = File(..., description="New audio file")
):
    """Start natural lip sync processing that preserves face features"""
    
    # Validate file types
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    
    video_ext = Path(input_video.filename).suffix.lower()
    audio_ext = Path(new_audio.filename).suffix.lower()
    
    if video_ext not in video_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported video format. Supported: {video_extensions}")
    
    if audio_ext not in audio_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format. Supported: {audio_extensions}")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded files
    video_path = f"uploads/{job_id}_video{video_ext}"
    audio_path = f"uploads/{job_id}_audio{audio_ext}"
    output_path = f"outputs/{job_id}_output.mp4"
    
    # Save files
    async with aiofiles.open(video_path, 'wb') as f:
        content = await input_video.read()
        await f.write(content)
    
    async with aiofiles.open(audio_path, 'wb') as f:
        content = await new_audio.read()
        await f.write(content)
    
    # Initialize job tracking
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Job queued for natural face-preserving processing',
        'created_at': datetime.now(),
        'video_path': video_path,
        'audio_path': audio_path,
        'output_path': output_path
    }
    
    # Start background processing
    background_tasks.add_task(
        pipeline.process_async,
        video_path,
        audio_path,
        output_path,
        job_id
    )
    
    return ProcessingResponse(
        job_id=job_id,
        status="queued",
        message="Job queued for natural face-preserving lip sync processing.",
        estimated_time="10-20 minutes"
    )

@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job['status'],
        progress=job.get('progress'),
        message=job['message'],
        output_url=job.get('output_url'),
        error=job.get('error')
    )

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download the processed video"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    output_path = job['output_path']
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type='video/mp4',
        filename=f"natural_lip_sync_{job_id}.mp4"
    )

@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up job files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    files_to_remove = [
        job.get('video_path'),
        job.get('audio_path'),
        job.get('output_path')
    ]
    
    removed_files = []
    for file_path in files_to_remove:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            removed_files.append(file_path)
    
    # Remove job from memory
    del jobs[job_id]
    
    return {"message": f"Cleaned up job {job_id}", "removed_files": removed_files}

@app.get("/jobs")
async def list_jobs():
    """List all jobs and their status"""
    job_list = []
    for job_id, job_data in jobs.items():
        job_list.append({
            "job_id": job_id,
            "status": job_data['status'],
            "progress": job_data.get('progress', 0),
            "message": job_data['message'],
            "created_at": job_data['created_at'].isoformat()
        })
    
    return {"jobs": job_list, "total": len(job_list)}

if __name__ == "__main__":
    uvicorn.run(
        "pipeline:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )