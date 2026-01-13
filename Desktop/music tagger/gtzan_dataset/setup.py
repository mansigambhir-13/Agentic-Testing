#!/usr/bin/env python3
"""
Quick Setup Script for GTZAN Music Genre Classification
========================================================
This script sets up the environment and runs initial tests
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60)

def check_python_version():
    """Check if Python version is suitable"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ is recommended")
        return False
    print("✓ Python version is suitable")
    return True

def check_kaggle_setup():
    """Check if Kaggle API is configured"""
    print_header("Checking Kaggle Setup")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("✓ Kaggle credentials found")
        
        # Check permissions on Unix-like systems
        if platform.system() != "Windows":
            import stat
            mode = kaggle_json.stat().st_mode
            if mode & stat.S_IROTH or mode & stat.S_IWOTH:
                print("⚠️  Warning: kaggle.json has overly permissive permissions")
                print("   Run: chmod 600 ~/.kaggle/kaggle.json")
        return True
    else:
        print("❌ Kaggle credentials not found")
        print("\nTo set up Kaggle API:")
        print("1. Sign up at https://www.kaggle.com")
        print("2. Go to Account → Create New API Token")
        print("3. Save kaggle.json to ~/.kaggle/")
        if platform.system() != "Windows":
            print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        else:
            print("4. On Windows, ensure the file is in: C:\\Users\\YourUsername\\.kaggle\\kaggle.json")
        return False

def create_virtual_env():
    """Create and activate virtual environment"""
    print_header("Virtual Environment Setup")
    
    response = input("\nCreate virtual environment? (recommended) [y/n]: ").lower()
    if response != 'y':
        print("Skipping virtual environment creation")
        return False
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("✓ Virtual environment already exists")
    else:
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✓ Virtual environment created")
    
    # Instructions for activation
    print("\nTo activate the virtual environment:")
    if platform.system() == "Windows":
        print("  Run: .\\venv\\Scripts\\activate")
    else:
        print("  Run: source venv/bin/activate")
    
    print("\n⚠️  Please activate the virtual environment and run this script again")
    return True

def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        print("Creating basic requirements.txt...")
        
        basic_requirements = """kagglehub>=0.1.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
"""
        with open("requirements.txt", "w") as f:
            f.write(basic_requirements)
    
    response = input("\nInstall all dependencies? [y/n]: ").lower()
    if response != 'y':
        print("Skipping dependency installation")
        return False
    
    print("\nInstalling packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("✓ Dependencies installed")
    return True

def test_imports():
    """Test if key imports work"""
    print_header("Testing Imports")
    
    packages = [
        ("kagglehub", "Kaggle Hub"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-learn"),
    ]
    
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name}")
            all_good = False
    
    # Optional packages
    print("\nOptional packages:")
    optional = [
        ("librosa", "Librosa (audio processing)"),
        ("tensorflow", "TensorFlow (deep learning)"),
        ("cv2", "OpenCV (image processing)"),
    ]
    
    for package, name in optional:
        try:
            if package == "cv2":
                import cv2
            else:
                __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"○ {name} - not installed")
    
    return all_good

def create_directories():
    """Create output directories"""
    print_header("Creating Output Directories")
    
    dirs = [
        "gtzan_visualizations",
        "gtzan_cnn_results",
        "models",
        "data"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created: {dir_name}/")

def run_quick_test():
    """Run a quick test to verify setup"""
    print_header("Running Quick Test")
    
    test_code = """
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Test basic operations
data = np.random.randn(100, 4)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
print(f"Created DataFrame with shape: {df.shape}")
# Test plotting
fig, ax = plt.subplots(figsize=(8, 4))
df.plot(ax=ax)
plt.title('Test Plot')
plt.close()
print("✓ Plotting works")
print("✓ All basic tests passed!")
"""
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("╔" + "=" * 58 + "╗")
    print("║" + " GTZAN Music Genre Classification Setup ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    print("\nThis script will help you set up the environment for")
    print("music genre classification with the GTZAN dataset.")
    
    # Run setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Kaggle Setup", check_kaggle_setup),
        ("Dependencies", install_dependencies),
        ("Import Test", test_imports),
        ("Directories", create_directories),
        ("Quick Test", run_quick_test),
    ]
    
    results = {}
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"Error in {step_name}: {e}")
            results[step_name] = False
    
    # Summary
    print_header("Setup Summary")
    all_good = True
    for step_name, success in results.items():
        status = "✓" if success else "❌"
        print(f"{status} {step_name}")
        if not success:
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Setup complete! You're ready to start.")
        print("\nNext steps:")
        print("1. Run: python gtzan_music_dataset.py")
        print("2. Run: python gtzan_cnn_classifier.py")
    else:
        print("⚠️  Some setup steps need attention.")
        print("Please fix the issues marked with ❌ above.")
    
    print("\nFor detailed instructions, see README.md")
    print("=" * 60)

if __name__ == "__main__":
    main()

