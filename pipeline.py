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
import importlib
import io
import json
from tqdm import tqdm
import tempfile
import shutil
import uuid
import threading
from datetime import datetime
from pathlib import Path
import asyncio
import aiofiles
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
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

class RealtimeSessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    ws_url: str
    fps: float
    sample_rate: int
    head_motion: bool
    head_motion_strength: float

# Global job storage
jobs = {}
realtime_sessions = {}
realtime_infer_lock = threading.Lock()
wav2lip_audio_lock = threading.Lock()
wav2lip_audio_module = None
tts_lock = threading.Lock()

def _first_existing_path(path_candidates):
    """Return the first path that exists, otherwise None."""
    for candidate in path_candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None

def _load_wav2lip_audio_module():
    """Load Wav2Lip audio.py module once and return it."""
    global wav2lip_audio_module
    if wav2lip_audio_module is not None:
        return wav2lip_audio_module

    with wav2lip_audio_lock:
        if wav2lip_audio_module is not None:
            return wav2lip_audio_module

        wav2lip_dir = str((Path(__file__).resolve().parent / "Wav2Lip").resolve())
        if wav2lip_dir not in sys.path:
            sys.path.insert(0, wav2lip_dir)

        wav2lip_audio_module = importlib.import_module("audio")
        return wav2lip_audio_module

def _patch_wav2lip_mel_basis_if_needed(audio_module):
    """Patch old Wav2Lip audio mel call signature for newer librosa versions."""
    hp = getattr(audio_module, "hp", None)
    if hp is None:
        raise RuntimeError("Wav2Lip audio module is missing hparams.")

    def _build_mel_basis_compat():
        assert hp.fmax <= hp.sample_rate // 2
        return librosa.filters.mel(
            sr=hp.sample_rate,
            n_fft=hp.n_fft,
            n_mels=hp.num_mels,
            fmin=hp.fmin,
            fmax=hp.fmax,
        )

    audio_module._build_mel_basis = _build_mel_basis_compat
    audio_module._mel_basis = None

def _wav2lip_melspectrogram(audio_module, wav):
    """Compute mels using Wav2Lip's native preprocessing."""
    try:
        return audio_module.melspectrogram(wav)
    except TypeError as exc:
        if "mel() takes 0 positional arguments" not in str(exc):
            raise
        _patch_wav2lip_mel_basis_if_needed(audio_module)
        return audio_module.melspectrogram(wav)

def _synthesize_tts_waveform(text, voice=None, rate=175):
    """Synthesize speech to mono float32 waveform at 16 kHz."""
    if not text or not text.strip():
        raise ValueError("Text for TTS is empty.")
    text = text.strip()

    with tts_lock:
        # macOS built-in TTS path.
        if shutil.which("say"):
            tmp_aiff = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False)
            tmp_aiff_path = tmp_aiff.name
            tmp_aiff.close()

            try:
                cmd = ["say"]
                if voice and str(voice).strip():
                    cmd += ["-v", str(voice).strip()]
                if rate:
                    cmd += ["-r", str(int(rate))]
                cmd += ["-o", tmp_aiff_path, text]

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip() or "macOS say command failed")

                wav, _ = librosa.load(tmp_aiff_path, sr=16000, mono=True)
                return np.asarray(wav, dtype=np.float32), 16000
            finally:
                if os.path.exists(tmp_aiff_path):
                    os.remove(tmp_aiff_path)

        # Optional Python fallback.
        try:
            import pyttsx3  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "No TTS backend available. Install pyttsx3 or use macOS built-in 'say' command."
            ) from exc

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()

        try:
            engine = pyttsx3.init()
            if rate:
                engine.setProperty("rate", int(rate))
            if voice and str(voice).strip():
                want = str(voice).strip().lower()
                selected = None
                for available in engine.getProperty("voices") or []:
                    name = (getattr(available, "name", "") or "").lower()
                    voice_id = (getattr(available, "id", "") or "").lower()
                    if want in name or want in voice_id:
                        selected = getattr(available, "id", None)
                        break
                if selected:
                    engine.setProperty("voice", selected)

            engine.save_to_file(text, tmp_wav_path)
            engine.runAndWait()

            wav, _ = librosa.load(tmp_wav_path, sr=16000, mono=True)
            return np.asarray(wav, dtype=np.float32), 16000
        finally:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)

def _waveform_to_wav_bytes(wav, sample_rate=16000):
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32), int(sample_rate), format="WAV", subtype="PCM_16")
    return buf.getvalue()

class LipSyncPipeline:
    def __init__(self, use_gan=False):
        """Initialize the lip sync pipeline in CPU-only mode"""
        self.device = 'cpu'
        self.use_gan = use_gan

        # Add Wav2Lip repo to path
        sys.path.append('Wav2Lip')

        # Select model paths with fallbacks for this repo layout.
        self.model_path_candidates = (
            ["models/Wav2Lip-SD-GAN.pt", "checkpoints/wav2lip_gan.pth"]
            if use_gan
            else ["checkpoints/wav2lip.pth", "models/Wav2Lip-SD-NOGAN.pt"]
        )
        self.s3fd_model_path_candidates = [
            "models/face_detection/s3fd-619a316812.pth",
            "Wav2Lip/face_detection/detection/sfd/s3fd.pth",
        ]

        self.model_path = None
        self.s3fd_model_path = None

        self.model = None
        self.face_detect = None
        self._model_loaded = False

    def load_models(self):
        """Load Wav2Lip and face detection models"""
        if self._model_loaded:
            return
            
        try:
            logger.info(f"Loading {'GAN' if self.use_gan else 'Non-GAN'} model on CPU...")

            self.model_path = _first_existing_path(self.model_path_candidates)
            if not self.model_path:
                raise FileNotFoundError(
                    f"Wav2Lip model not found. Tried: {', '.join(self.model_path_candidates)}"
                )

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
            self.s3fd_model_path = _first_existing_path(self.s3fd_model_path_candidates)
            if not self.s3fd_model_path:
                raise FileNotFoundError(
                    f"S3FD model not found. Tried: {', '.join(self.s3fd_model_path_candidates)}"
                )

            self.face_detect = s3fd()
            self.face_detect.load_state_dict(torch.load(self.s3fd_model_path, map_location='cpu', weights_only=False))
            self.face_detect.eval()

            self._model_loaded = True
            logger.info("All models loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(traceback.format_exc())
            raise

    def detect_primary_face_bbox(self, frame, pads=(0, 10, 0, 0)):
        """Detect the largest face in a single frame and return (y1, y2, x1, x2)."""
        if self.s3fd_model_path is None:
            self.s3fd_model_path = _first_existing_path(self.s3fd_model_path_candidates)
        if self.s3fd_model_path is None:
            raise FileNotFoundError("Could not locate S3FD detector weights.")

        from Wav2Lip.face_detection.detection.sfd.sfd_detector import SFDDetector

        detector = SFDDetector(device='cpu', path_to_detector=self.s3fd_model_path)
        faces = detector.detect_from_image(frame)
        if not faces:
            raise ValueError("No face detected in the provided avatar image.")

        face_bbox = max(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        x1, y1, x2, y2 = map(int, face_bbox[:4])
        top, bottom, left, right = [int(v) for v in pads]
        h, w = frame.shape[:2]
        y1 = max(0, y1 - top)
        y2 = min(h, y2 + bottom)
        x1 = max(0, x1 - left)
        x2 = min(w, x2 + right)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Detected face bbox is invalid after applying pads.")
        return (y1, y2, x1, x2)

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
            detect_face = SFDDetector(device='cpu', path_to_detector=self.s3fd_model_path)
            
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

class RealtimeStaticLipSyncSession:
    """CPU-only real-time lip-sync session for a static avatar image."""

    def __init__(
        self,
        session_id,
        pipeline_obj,
        frame_bgr,
        face_bbox,
        fps=15.0,
        input_sample_rate=16000,
        enable_head_motion=True,
        head_motion_strength=0.55,
    ):
        self.session_id = session_id
        self.pipeline = pipeline_obj
        self.model = pipeline_obj.model
        self.audio_module = _load_wav2lip_audio_module()

        self.frame = frame_bgr
        self.face_bbox = face_bbox
        self.fps = float(fps)

        self.target_sample_rate = 16000
        self.input_sample_rate = int(input_sample_rate)

        self.mel_step_size = 16
        self.hop_length = int(getattr(self.audio_module.hp, "hop_size", 200) or 200)
        self.n_fft = int(getattr(self.audio_module.hp, "n_fft", 800) or 800)
        self.min_audio_for_mel = max(self.n_fft, self.hop_length)
        self.mel_idx_multiplier = 80.0 / max(self.fps, 1.0)

        # Keep only a short rolling window to bound CPU cost.
        self.max_audio_samples = self.target_sample_rate * 6
        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.next_mel_start = 0.0
        self.frames_sent = 0

        self.enable_head_motion = bool(enable_head_motion)
        self.head_motion_strength = float(np.clip(head_motion_strength, 0.0, 1.5))
        self.smoothed_energy = 0.0

        self._state_lock = threading.Lock()
        self._face_input_tensor = self._build_face_input_tensor(self.frame, self.face_bbox)
        self._motion_mask = self._build_face_motion_mask()
        self._motion_rng = np.random.default_rng(abs(hash(self.session_id)) % (2**32))
        self._phase_yaw = float(self._motion_rng.uniform(0.0, 2.0 * np.pi))
        self._phase_pitch = float(self._motion_rng.uniform(0.0, 2.0 * np.pi))
        self._phase_roll = float(self._motion_rng.uniform(0.0, 2.0 * np.pi))
        self._phase_x = float(self._motion_rng.uniform(0.0, 2.0 * np.pi))

    def _build_face_input_tensor(self, frame, face_bbox):
        y1, y2, x1, x2 = face_bbox
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (96, 96), interpolation=cv2.INTER_AREA)
        img = face.astype(np.float32) / 255.0
        masked = img.copy()
        masked[48:, :, :] = 0.0
        stacked = np.concatenate([masked, img], axis=2)
        return torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).float()

    def _build_face_motion_mask(self):
        y1, y2, x1, x2 = self.face_bbox
        h, w = self.frame.shape[:2]
        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * 0.45)
        pad_y = int(bh * 0.55)

        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(w, x2 + pad_x)
        ry2 = min(h, y2 + pad_y)

        roi_h = ry2 - ry1
        roi_w = rx2 - rx1
        if roi_h <= 0 or roi_w <= 0:
            return None

        mask = np.zeros((roi_h, roi_w), dtype=np.float32)
        center = (roi_w // 2, roi_h // 2)
        axes = (max(1, int(roi_w * 0.40)), max(1, int(roi_h * 0.45)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        blur_kernel = max(3, (min(roi_h, roi_w) // 12) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        return {
            "roi": (ry1, ry2, rx1, rx2),
            "mask": np.expand_dims(mask, axis=2),
        }

    def set_input_sample_rate(self, sample_rate):
        if sample_rate and int(sample_rate) > 0:
            if int(sample_rate) != self.input_sample_rate:
                self.reset()
            self.input_sample_rate = int(sample_rate)

    def set_head_motion(self, enabled=None, strength=None):
        if enabled is not None:
            self.enable_head_motion = bool(enabled)
        if strength is not None:
            self.head_motion_strength = float(np.clip(float(strength), 0.0, 1.5))

    def set_fps(self, fps):
        fps_val = float(fps)
        fps_val = float(np.clip(fps_val, 5.0, 30.0))
        if abs(fps_val - self.fps) < 1e-6:
            return
        self.fps = fps_val
        self.mel_idx_multiplier = 80.0 / max(self.fps, 1.0)

    def reset(self):
        with self._state_lock:
            self.audio_buffer = np.zeros((0,), dtype=np.float32)
            self.next_mel_start = 0.0
            self.frames_sent = 0

    def _resample_to_target(self, audio_chunk):
        if self.input_sample_rate == self.target_sample_rate:
            return audio_chunk.astype(np.float32, copy=False)
        if audio_chunk.size == 0:
            return audio_chunk.astype(np.float32, copy=False)
        resampled = librosa.resample(
            audio_chunk.astype(np.float32, copy=False),
            orig_sr=self.input_sample_rate,
            target_sr=self.target_sample_rate,
        )
        return np.asarray(resampled, dtype=np.float32)

    def _trim_audio_buffer_if_needed(self):
        if self.audio_buffer.size <= self.max_audio_samples:
            return
        drop_samples = int(self.audio_buffer.size - self.max_audio_samples)
        self.audio_buffer = self.audio_buffer[drop_samples:]
        dropped_mel_frames = float(drop_samples) / float(self.hop_length)
        self.next_mel_start = max(0.0, self.next_mel_start - dropped_mel_frames)

    def _next_mel_chunk_from_mel(self, mel):
        if mel is None or mel.ndim != 2 or mel.shape[1] < self.mel_step_size:
            return None

        start_idx = int(self.next_mel_start)
        if start_idx + self.mel_step_size > mel.shape[1]:
            return None

        chunk = mel[:, start_idx:start_idx + self.mel_step_size]
        self.next_mel_start += self.mel_idx_multiplier
        return chunk

    def _estimate_audio_energy(self, mel_chunk):
        energy = float(np.mean(np.abs(mel_chunk)) / 4.0)
        energy = float(np.clip(energy, 0.0, 1.0))
        self.smoothed_energy = 0.88 * self.smoothed_energy + 0.12 * energy
        return self.smoothed_energy

    def _apply_head_motion(self, frame, energy):
        if not self.enable_head_motion or self._motion_mask is None:
            return frame

        y1, y2, x1, x2 = self._motion_mask["roi"]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        t = float(self.frames_sent) / max(self.fps, 1.0)
        gain = self.head_motion_strength * (0.25 + 0.75 * float(np.clip(energy, 0.0, 1.0)))

        x_shift = 2.0 * gain * np.sin((2.0 * np.pi * 0.52 * t) + self._phase_x)
        y_shift = 2.7 * gain * np.sin((2.0 * np.pi * 0.43 * t) + self._phase_pitch)
        roll_deg = 1.2 * gain * np.sin((2.0 * np.pi * 0.37 * t) + self._phase_roll)
        yaw_like_deg = 1.6 * gain * np.sin((2.0 * np.pi * 0.61 * t) + self._phase_yaw)
        angle = roll_deg + 0.35 * yaw_like_deg

        center = (roi.shape[1] * 0.5, roi.shape[0] * 0.5)
        transform = cv2.getRotationMatrix2D(center, angle, 1.0)
        transform[0, 2] += x_shift
        transform[1, 2] += y_shift

        moved_roi = cv2.warpAffine(
            roi,
            transform,
            (roi.shape[1], roi.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = self._motion_mask["mask"]
        blended = (moved_roi.astype(np.float32) * mask) + (roi.astype(np.float32) * (1.0 - mask))
        out = frame.copy()
        out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def _infer_one_frame(self, mel_chunk):
        mel_tensor = torch.from_numpy(mel_chunk.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        with realtime_infer_lock:
            with torch.no_grad():
                pred = self.model(mel_tensor, self._face_input_tensor)

        patch = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        patch = np.clip(patch * 255.0, 0, 255).astype(np.uint8)

        y1, y2, x1, x2 = self.face_bbox
        patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        out = self.frame.copy()
        out[y1:y2, x1:x2] = patch
        out = self._apply_head_motion(out, self._estimate_audio_energy(mel_chunk))
        return out

    @staticmethod
    def _encode_jpeg(frame_bgr, quality=80):
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("Failed to encode generated frame.")
        return encoded.tobytes()

    def consume_pcm_f32le_bytes(self, audio_bytes):
        """
        Consume little-endian float32 mono PCM bytes and return generated JPEG frames.
        """
        if not audio_bytes:
            return []

        raw = np.frombuffer(audio_bytes, dtype="<f4")
        if raw.size == 0:
            return []

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        audio_chunk = self._resample_to_target(raw)
        if audio_chunk.size == 0:
            return []

        with self._state_lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            self._trim_audio_buffer_if_needed()
            if self.audio_buffer.size < self.min_audio_for_mel:
                return []

            mel = _wav2lip_melspectrogram(self.audio_module, self.audio_buffer)
            if np.isnan(mel).any():
                logger.warning(f"Realtime mel contains NaN values; dropping chunk (session={self.session_id})")
                return []

            jpeg_frames = []
            while True:
                mel_chunk = self._next_mel_chunk_from_mel(mel)
                if mel_chunk is None:
                    break
                frame = self._infer_one_frame(mel_chunk)
                jpeg_frames.append(self._encode_jpeg(frame))
                self.frames_sent += 1

            return jpeg_frames

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
    return FileResponse(
        "static/index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.post("/realtime/session", response_model=RealtimeSessionResponse)
async def create_realtime_session(
    request: Request,
    avatar_image: UploadFile = File(..., description="Static avatar image (jpg/png/jpeg)"),
    fps: float = Form(15.0),
    input_sample_rate: int = Form(16000),
    head_motion: bool = Form(True),
    head_motion_strength: float = Form(0.55),
):
    """Create a CPU-only real-time lip-sync session for a static avatar image."""
    if fps < 5.0 or fps > 30.0:
        raise HTTPException(status_code=400, detail="fps must be between 5 and 30.")
    if input_sample_rate <= 0:
        raise HTTPException(status_code=400, detail="input_sample_rate must be > 0.")
    if head_motion_strength < 0.0 or head_motion_strength > 1.5:
        raise HTTPException(status_code=400, detail="head_motion_strength must be between 0.0 and 1.5.")

    img_ext = Path(avatar_image.filename or "").suffix.lower()
    if img_ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported avatar image format.")

    avatar_bytes = await avatar_image.read()
    if not avatar_bytes:
        raise HTTPException(status_code=400, detail="Avatar image payload is empty.")

    avatar_np = np.frombuffer(avatar_bytes, dtype=np.uint8)
    frame = cv2.imdecode(avatar_np, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode avatar image.")

    # Keep avatar size bounded for CPU throughput.
    h, w = frame.shape[:2]
    max_side = 768
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    try:
        await asyncio.to_thread(pipeline.load_models)
        face_bbox = await asyncio.to_thread(pipeline.detect_primary_face_bbox, frame)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to initialize real-time session: {exc}") from exc

    session_id = str(uuid.uuid4())
    session = RealtimeStaticLipSyncSession(
        session_id=session_id,
        pipeline_obj=pipeline,
        frame_bgr=frame,
        face_bbox=face_bbox,
        fps=float(fps),
        input_sample_rate=int(input_sample_rate),
        enable_head_motion=bool(head_motion),
        head_motion_strength=float(head_motion_strength),
    )
    realtime_sessions[session_id] = session

    ws_scheme = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{request.url.netloc}/ws/realtime/{session_id}"

    return RealtimeSessionResponse(
        session_id=session_id,
        status="ready",
        message="Real-time session created. Stream float32 PCM mono chunks over WebSocket.",
        ws_url=ws_url,
        fps=float(fps),
        sample_rate=session.target_sample_rate,
        head_motion=session.enable_head_motion,
        head_motion_strength=session.head_motion_strength,
    )

@app.post("/realtime/tts")
async def realtime_tts(
    text: str = Form(...),
    voice: Optional[str] = Form(None),
    rate: int = Form(175),
):
    """Synthesize text to WAV audio for realtime avatar feeding."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    if rate < 80 or rate > 320:
        raise HTTPException(status_code=400, detail="rate must be between 80 and 320 words per minute.")

    try:
        wav, sr = await asyncio.to_thread(_synthesize_tts_waveform, text.strip(), voice, int(rate))
        wav_bytes = await asyncio.to_thread(_waveform_to_wav_bytes, wav, sr)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

    return Response(content=wav_bytes, media_type="audio/wav")

@app.delete("/realtime/session/{session_id}")
async def delete_realtime_session(session_id: str):
    """Delete an active real-time session."""
    session = realtime_sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="Real-time session not found.")
    return {"session_id": session_id, "status": "deleted"}

@app.websocket("/ws/realtime/{session_id}")
async def realtime_stream(websocket: WebSocket, session_id: str):
    """Real-time stream: receive audio chunks (binary float32 PCM), send generated JPEG frames (binary)."""
    session = realtime_sessions.get(session_id)
    if session is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    await websocket.send_json({
        "type": "ready",
        "session_id": session_id,
        "fps": session.fps,
        "sample_rate": session.target_sample_rate,
        "head_motion": session.enable_head_motion,
        "head_motion_strength": session.head_motion_strength,
    })

    try:
        while True:
            message = await websocket.receive()

            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                break

            raw_bytes = message.get("bytes")
            if raw_bytes is not None:
                try:
                    jpeg_frames = await asyncio.to_thread(session.consume_pcm_f32le_bytes, raw_bytes)
                except Exception as exc:
                    logger.error(f"Real-time frame generation failed (session={session_id}): {exc}")
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

                for frame_bytes in jpeg_frames:
                    await websocket.send_bytes(frame_bytes)
                continue

            text_payload = message.get("text")
            if text_payload is None:
                continue

            try:
                payload = json.loads(text_payload)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON control message."})
                continue

            command = payload.get("type")
            if command == "ping":
                await websocket.send_json({"type": "pong", "frames_sent": session.frames_sent})
            elif command == "config":
                new_sr = int(payload.get("input_sample_rate", session.input_sample_rate))
                session.set_input_sample_rate(new_sr)
                if "fps" in payload:
                    session.set_fps(payload.get("fps", session.fps))
                if "head_motion" in payload or "head_motion_strength" in payload:
                    session.set_head_motion(
                        enabled=payload.get("head_motion", None),
                        strength=payload.get("head_motion_strength", None),
                    )
                await websocket.send_json({
                    "type": "config_ack",
                    "input_sample_rate": session.input_sample_rate,
                    "fps": session.fps,
                    "head_motion": session.enable_head_motion,
                    "head_motion_strength": session.head_motion_strength,
                })
            elif command == "reset":
                session.reset()
                await websocket.send_json({"type": "reset_ack"})
            elif command == "close":
                await websocket.close(code=1000)
                break
            else:
                await websocket.send_json({"type": "error", "message": "Unsupported control command."})

    except WebSocketDisconnect:
        logger.info(f"Real-time websocket disconnected: {session_id}")
    except Exception as exc:
        logger.error(f"Real-time websocket error (session={session_id}): {exc}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        realtime_sessions.pop(session_id, None)

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
        host="localhost",
        port=8000,
        reload=True
    )
