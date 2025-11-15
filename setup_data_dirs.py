"""Script to create data directory structure."""
from pathlib import Path

def create_data_structure(base_dir="data"):
    """Create data directory structure."""
    base = Path(base_dir)
    
    dirs = [
        base / "train" / "real",
        base / "train" / "fake",
        base / "val" / "real",
        base / "val" / "fake",
        base / "test" / "real",
        base / "test" / "fake",
        base / "face_cache"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
    
    # Create README in data directory
    readme_content = """# Data Directory

Place your training, validation, and test data in the respective subdirectories.

## Structure:
- `train/real/` - Real training videos/images
- `train/fake/` - Fake training videos/images
- `val/real/` - Real validation videos/images
- `val/fake/` - Fake validation videos/images
- `test/real/` - Real test videos/images
- `test/fake/` - Fake test videos/images

## Supported Formats:
- Videos: .mp4, .avi, .mov, .mkv
- Images: .jpg, .jpeg, .png

## Recommended Datasets:
- FaceForensics++
- DFDC (DeepFake Detection Challenge)
- Celeb-DF
- FaceSwap / Face2Face / NeuralTextures
"""
    
    readme_path = base / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\nCreated README at: {readme_path}")
    print("\nData directory structure created successfully!")
    print("Please add your data files to the respective directories.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create data directory structure')
    parser.add_argument('--data_dir', type=str, default='data', help='Base data directory')
    args = parser.parse_args()
    
    create_data_structure(args.data_dir)

