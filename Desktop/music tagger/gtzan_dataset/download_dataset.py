"""
Quick script to download GTZAN dataset
"""
import sys
from pathlib import Path

try:
    import kagglehub
    
    print("=" * 60)
    print("Downloading GTZAN Music Genre Classification Dataset")
    print("=" * 60)
    print("\nThis may take a few minutes (dataset is ~1.2GB)...")
    print()
    
    # Download dataset
    dataset_path = kagglehub.dataset_download(
        "andradaolteanu/gtzan-dataset-music-genre-classification"
    )
    
    dataset_path = Path(dataset_path)
    print(f"\n[OK] Dataset downloaded successfully!")
    print(f"[OK] Location: {dataset_path}")
    print(f"\nDataset structure:")
    
    # List contents
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            file_count = len(list(item.rglob("*")))
            print(f"  [DIR] {item.name}/ ({file_count} items)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  [FILE] {item.name} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("Download complete! You can now run:")
    print("  python gtzan_music_dataset.py")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check that kaggle.json is in: C:\\Users\\LENOVO\\.kaggle\\")
    print("2. Verify your Kaggle credentials are correct")
    print("3. Make sure you've accepted the dataset terms on Kaggle")
    print("4. Check your internet connection")
    sys.exit(1)

