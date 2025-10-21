import numpy as np
import cv2
import wave
import struct
import os

# Create a sample video file
def create_sample_video(output_path, duration=5, fps=30):
    # Video settings
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Generate frames
    for _ in range(duration * fps):
        # Create a frame with gray background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        # Draw an oval face shape that looks realistic
        center = (width // 2, height // 2)
        axes = (100, 130)  # Wider vertically for a more natural face shape
        
        # Face oval
        cv2.ellipse(frame, center, axes, 0, 0, 360, (220, 213, 203), -1)  # Fill with skin tone
        cv2.ellipse(frame, center, axes, 0, 0, 360, (192, 186, 178), 2)   # Darker outline
        
        # Eyes
        eye_color = (255, 255, 255)  # White for eyes
        iris_color = (72, 87, 150)   # Brown for iris
        
        # Left eye
        cv2.ellipse(frame, (center[0] - 40, center[1] - 20), (25, 15), 0, 0, 360, eye_color, -1)
        cv2.ellipse(frame, (center[0] - 40, center[1] - 20), (25, 15), 0, 0, 360, (100, 100, 100), 1)
        cv2.circle(frame, (center[0] - 40, center[1] - 20), 8, iris_color, -1)
        cv2.circle(frame, (center[0] - 40, center[1] - 20), 3, (0, 0, 0), -1)
        
        # Right eye
        cv2.ellipse(frame, (center[0] + 40, center[1] - 20), (25, 15), 0, 0, 360, eye_color, -1)
        cv2.ellipse(frame, (center[0] + 40, center[1] - 20), (25, 15), 0, 0, 360, (100, 100, 100), 1)
        cv2.circle(frame, (center[0] + 40, center[1] - 20), 8, iris_color, -1)
        cv2.circle(frame, (center[0] + 40, center[1] - 20), 3, (0, 0, 0), -1)
        
        # Nose
        nose_points = np.array([[center[0], center[1] - 5],
                              [center[0] - 15, center[1] + 15],
                              [center[0] + 15, center[1] + 15]], np.int32)
        cv2.polylines(frame, [nose_points], True, (160, 150, 140), 2)
        
        # Mouth
        mouth_color = (150, 100, 100)
        cv2.ellipse(frame, (center[0], center[1] + 40), (40, 20), 0, 0, 180, mouth_color, 3)
        
        out.write(frame)
    
    out.release()

# Create a sample audio file
def create_sample_audio(output_path, duration=5, sample_rate=44100):
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Scale to 16-bit range and convert to integers
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(output_path, 'w') as wav_file:
        # Set parameters
        nchannels = 1
        sampwidth = 2  # 16-bit
        
        # Set wav parameters
        wav_file.setnchannels(nchannels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        
        # Write data
        wav_file.writeframes(audio_data.tobytes())

if __name__ == "__main__":
    # Create sample_data directory if it doesn't exist
    os.makedirs("sample_data", exist_ok=True)
    
    # Generate sample files
    create_sample_video("sample_data/input_face.mp4")
    create_sample_audio("sample_data/input_audio.wav")
    
    print("Sample files generated successfully!")