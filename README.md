# Audio Replacement with Lip Sync

This project implements an audio replacement system with lip synchronization using Wav2Lip. It takes a video of a person speaking and replaces the original audio with new audio while modifying the lip movements to match the new speech.

## Overview

The system performs the following operations:
1. Extracts frames from the input video
2. Processes the new audio to match video duration
3. Uses Wav2Lip model to generate lip-synced frames
4. Combines the processed frames with new audio into output video

## Requirements

### System Requirements
- Python 3.7 or higher
- GPU recommended (CUDA-compatible) for faster processing
- At least 4GB RAM (8GB recommended)
- FFmpeg installed on system

### Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

### Additional Requirements
1. **FFmpeg**: Required for video processing
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **Wav2Lip Repository**: Clone the original repository
   ```bash
   git clone https://github.com/Rudrabha/Wav2Lip.git
   ```

3. **Model Files**: Download pre-trained models
   ```bash
   python download_models.py
   ```

## Installation

1. **Clone this repository:**
   ```bash
   git clone <your-repo-url>
   cd lip-sync-assignment
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Clone Wav2Lip repository:**
   ```bash
   git clone https://github.com/Rudrabha/Wav2Lip.git
   ```

5. **Download models:**
   ```bash
   python download_models.py
   ```

## Usage

### Basic Usage
```bash
python lip_sync_pipeline.py --input_video data/input_video.mp4 --new_audio data/new_audio.wav --output output/output_video.mp4
```

### Command Line Arguments
- `--input_video`: Path to input video file (required)
- `--new_audio`: Path to new audio file (required)
- `--output`: Path to output video file (required)
- `--model_path`: Path to Wav2Lip model file (default: models/wav2lip_gan.pth)

### Example with Custom Model Path
```bash
python lip_sync_pipeline.py \
  --input_video data/input_video.mp4 \
  --new_audio data/new_audio.wav \
  --output output/result.mp4 \
  --model_path custom_models/wav2lip_gan.pth
```

## Project Structure
```
lip-sync-assignment/
├── README.md                 # This file
├── lip_sync_pipeline.py     # Main pipeline script
├── download_models.py       # Script to download models
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── models/                 # Model files (created after setup)
│   └── wav2lip_gan.pth    # Wav2Lip pre-trained model
├── data/                   # Input data directory
│   ├── input_video.mp4    # Your input video (Paragraph A)
│   └── new_audio.wav      # Your new audio (Paragraph B)
├── output/                 # Output directory
│   └── output_video.mp4   # Generated result
└── Wav2Lip/               # Cloned Wav2Lip repository
```

## Input Requirements

### Input Video (`input_video.mp4`)
- **Duration**: 10-15 seconds
- **Content**: You speaking Paragraph A clearly
- **Quality**: Good lighting, clear face visibility
- **Format**: MP4, AVI, or MOV
- **Resolution**: Minimum 480p recommended

**Paragraph A:**
> "Today, I'm going to show you how machine learning can transform everyday tasks. Whether it's identifying objects in images or generating text from scratch, the possibilities are truly endless. Let's dive into some amazing examples together."

### New Audio (`new_audio.wav`)
- **Content**: You speaking Paragraph B clearly
- **Format**: WAV, MP3, or other common audio formats
- **Quality**: Clear speech, minimal background noise
- **Sample Rate**: Any (will be resampled to 16kHz)

**Paragraph B:**
> "The quick brown fox jumps over the lazy dog while a curious cat watches from the rooftop. Suddenly, a loud bark sends the animals running in every direction — pure chaos ensues."

## Expected Output

The system generates `output_video.mp4` with the following characteristics:
- **Video**: Original visual content with modified lip movements
- **Audio**: Replaced with new audio content (Paragraph B)
- **Synchronization**: Lip movements match the new audio
- **Quality**: Maintains original video resolution and lighting
- **Identity**: Preserves facial features and expressions

## Technical Details

### Core Components
1. **Frame Extraction**: Uses OpenCV to extract video frames
2. **Audio Processing**: Uses librosa for audio preprocessing
3. **Face Detection**: Detects and tracks faces in video frames
4. **Lip Sync Generation**: Uses Wav2Lip model for lip movement generation
5. **Video Composition**: Combines processed frames with new audio

### Processing Pipeline
1. Load pre-trained Wav2Lip model
2. Extract frames from input video at original FPS
3. Preprocess new audio (resample, trim/extend to match video duration)
4. For each frame:
   - Detect face region
   - Generate mel-spectrogram chunk for corresponding audio
   - Apply Wav2Lip model to generate lip-synced face
   - Blend result back into original frame
5. Combine all frames with new audio using FFmpeg

## Performance Notes

### Processing Times (Approximate)
- **GPU (RTX 3080)**: 2-5 minutes for 15-second video
- **GPU (GTX 1060)**: 5-10 minutes for 15-second video
- **CPU (Intel i7)**: 15-30 minutes for 15-second video

### Memory Requirements
- **GPU**: 2-4GB VRAM
- **RAM**: 4-8GB system memory
- **Storage**: ~500MB for models + temporary files

## Assumptions and Constraints

### Assumptions
1. Input video contains a single person speaking
2. Face is clearly visible throughout the video
3. Audio quality is clear and audible
4. Video has reasonable lighting conditions
5. Speaker is looking generally toward the camera

### Current Limitations
1. **Single Speaker**: Designed for single-person videos
2. **Face Visibility**: Requires clear face detection
3. **Processing Time**: Can be slow on CPU-only systems
4. **Quality**: Output quality depends on input quality
5. **Extreme Poses**: May struggle with extreme head angles

### Potential Improvements
1. **Multi-face Support**: Handle multiple speakers
2. **Better Blending**: Improve face blending techniques
3. **Quality Enhancement**: Add post-processing for better quality
4. **Real-time Processing**: Optimize for faster processing
5. **Expression Preservation**: Better preserve original expressions

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'Wav2Lip'"**
   - Solution: Ensure Wav2Lip repository is cloned in the project directory
   - Run: `git clone https://github.com/Rudrabha/Wav2Lip.git`

2. **"FileNotFoundError: Model file not found"**
   - Solution: Download the model file
   - Run: `python download_models.py`

3. **"CUDA out of memory" error**
   - Solution: Use CPU processing by setting appropriate device
   - Or reduce video resolution/duration

4. **"FFmpeg not found" error**
   - Solution: Install FFmpeg on your system
   - Follow installation instructions above

5. **Poor lip sync quality**
   - Check input video quality (lighting, face clarity)
   - Ensure audio is clear and matches expected duration
   - Try with different face angles in input video

### Performance Optimization
1. **Use GPU**: Ensure CUDA is properly installed for GPU acceleration
2. **Reduce Resolution**: Lower input video resolution for faster processing
3. **Shorter Videos**: Process shorter segments for quicker results
4. **Close Other Applications**: Free up system resources

## Development Notes

### Code Structure
- **Modular Design**: Each processing step is in separate methods
- **Error Handling**: Includes basic error handling and validation
- **Configurable**: Command-line arguments for flexibility
- **Temporary Files**: Proper cleanup of temporary files

### Future Enhancements
1. **GUI Interface**: Add graphical user interface
2. **Batch Processing**: Process multiple videos simultaneously
3. **Quality Metrics**: Add automated quality assessment
4. **Advanced Blending**: Implement Poisson blending or similar techniques
5. **Real-time Preview**: Add preview functionality during processing

## License

This project uses the Wav2Lip model which has its own license terms. Please refer to the original Wav2Lip repository for licensing information.

## References

- [Wav2Lip: Accurately Lip-syncing Videos to Any Speech](https://github.com/Rudrabha/Wav2Lip)
- [Original Wav2Lip Paper](https://arxiv.org/abs/2008.10010)
- [Face Alignment Library](https://github.com/1adrianb/face-alignment)

## Contact

For questions about this implementation, please refer to the assignment guidelines or contact the interviewer.# wav2lip-audio-replacement
A Python-based lip-syncing pipeline that replaces original audio in a video with new audio and syncs lip movements using Wav2Lip.


This project demonstrates an audio replacement system with accurate lip synchronization using the Wav2Lip model. It takes an input video and a new audio file, then generates an output video where the original speech is replaced with the new audio while ensuring the speaker’s lip movements match the new speech. The system maintains the original video’s identity, head movements, and visual quality for a realistic result.
