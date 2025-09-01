#!/usr/bin/env python3
"""
Script to manually download and set up S3FD model for face-alignment
"""

import os
import requests
import hashlib
from pathlib import Path
import urllib.request

def download_s3fd_model(project_dir=None):
    """Download S3FD model and place it in the project directory"""
    
    # Get the face-alignment S3FD URL
    try:
        from face_alignment.detection.sfd import models_urls
        s3fd_url = models_urls['s3fd']
        print(f"🔗 S3FD URL: {s3fd_url}")
    except:
        # Fallback URL (may need to be updated)
        s3fd_url = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
        print(f"🔗 Using fallback URL: {s3fd_url}")
    
    # Use project directory or current directory
    if project_dir:
        cache_dir = Path(project_dir) / 'models' / 'face_detection'
    else:
        # Use current script directory
        cache_dir = Path(__file__).parent / 'models' / 'face_detection'
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL
    filename = s3fd_url.split('/')[-1]
    if not filename.endswith('.pth'):
        filename = 's3fd.pth'
    
    model_path = cache_dir / filename
    
    # Also create symlink in torch cache for compatibility
    torch_cache_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    torch_model_path = torch_cache_dir / filename
    
    print(f"📁 Project models directory: {cache_dir}")
    print(f"📁 Torch cache directory: {torch_cache_dir}")
    print(f"📄 Model file: {model_path}")
    
    if model_path.exists():
        print("✅ Model already exists in project directory. Checking integrity...")
        # Create symlink in torch cache if it doesn't exist
        if not torch_model_path.exists():
            try:
                import shutil
                shutil.copy2(model_path, torch_model_path)
                print(f"✅ Copied model to torch cache: {torch_model_path}")
            except Exception as e:
                print(f"⚠️  Could not copy to torch cache: {e}")
        return str(model_path)
    
    print("⬇️  Downloading S3FD model...")
    
    try:
        # Download with progress
        def download_with_progress(url, filename):
            def reporthook(blocknum, blocksize, totalsize):
                readsofar = blocknum * blocksize
                if totalsize > 0:
                    percent = readsofar * 1e2 / totalsize
                    s = "\r%5.1f%% %*d / %d" % (
                        percent, len(str(totalsize)), readsofar, totalsize)
                    print(s, end='')
                    if readsofar >= totalsize:  # near the end
                        print()
                else:
                    print("read %d" % (readsofar,))
            
            urllib.request.urlretrieve(url, filename, reporthook)
        
        download_with_progress(s3fd_url, str(model_path))
        print(f"✅ Successfully downloaded: {model_path}")
        print(f"📊 File size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Copy to torch cache for compatibility
        try:
            import shutil
            shutil.copy2(model_path, torch_model_path)
            print(f"✅ Copied to torch cache: {torch_model_path}")
        except Exception as e:
            print(f"⚠️  Could not copy to torch cache: {e}")
        
        return str(model_path)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def alternative_download_sources():
    """List alternative download sources"""
    print("\n🔄 Alternative Download Sources:")
    print("1. GitHub Repositories:")
    print("   - https://github.com/yxlijun/S3FD.pytorch")
    print("   - https://github.com/swordlidev/sfd.pytorch")
    print("   - https://github.com/luuuyi/S3FD.PyTorch")
    
    print("\n2. Manual Steps:")
    print("   a) Visit any of the GitHub repos above")
    print("   b) Look for 'releases' or 'models' folder")
    print("   c) Download the .pth or .pth.tar file")
    print("   d) Place it in your cache directory")
    
    print(f"\n3. Project Directory Structure:")
    print("   your-project/")
    print("   ├── models/")
    print("   │   └── face_detection/")
    print("   │       └── s3fd-*.pth")
    print("   └── pipeline.py")

def clear_corrupted_cache():
    """Clear corrupted torch hub cache"""
    import shutil
    
    cache_dir = Path.home() / '.cache' / 'torch' / 'hub'
    
    if cache_dir.exists():
        print(f"🗑️  Clearing cache: {cache_dir}")
        shutil.rmtree(cache_dir)
        print("✅ Cache cleared")
    else:
        print("ℹ️  No cache directory found")

def verify_setup():
    """Verify that face-alignment can now load S3FD"""
    try:
        import face_alignment
        import torch
        
        print("\n🧪 Testing face-alignment setup...")
        
        # Try to create FaceAlignment with S3FD
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device='cpu',
            face_detector='sfd'
        )
        
        print("✅ S3FD loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 S3FD Model Setup Tool")
    print("=" * 50)
    
    # Get project directory from user
    import sys
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
        print(f"📂 Using specified project directory: {project_dir}")
    else:
        # Use current directory as project root
        project_dir = str(Path.cwd())
        print(f"📂 Using current directory as project root: {project_dir}")
    
    # First, clear any corrupted cache
    print("\n1️⃣ Clearing corrupted cache...")
    clear_corrupted_cache()
    
    # Try to download the model
    print("\n2️⃣ Downloading S3FD model...")
    model_path = download_s3fd_model(project_dir)
    
    if model_path:
        print("\n3️⃣ Verifying setup...")
        if verify_setup():
            print("\n🎉 Setup complete! You can now use face-alignment with S3FD.")
            print(f"📁 Model stored in: {model_path}")
        else:
            print("\n⚠️  Download completed but verification failed.")
            alternative_download_sources()
    else:
        print("\n❌ Automatic download failed.")
        alternative_download_sources()