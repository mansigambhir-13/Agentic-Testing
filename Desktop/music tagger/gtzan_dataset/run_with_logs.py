"""
Run Classifier with Live Logs
==============================
Run the classifier and show live output logs.
"""

import sys
import subprocess
from pathlib import Path

def run_with_logs():
    """Run classifier with live log output"""
    script_path = Path("bedrock_music_classifier.py")
    
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return
    
    print("=" * 60)
    print(" Starting Music Genre Classifier with Live Logs ".center(60))
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    try:
        # Run the script and show output in real-time
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output line by line
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        
        if process.returncode == 0:
            print("\n" + "=" * 60)
            print(" Classification Complete! ".center(60))
            print("=" * 60)
        else:
            print(f"\nProcess exited with code {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n\nStopping classifier...")
        process.terminate()
        process.wait()
        print("Classifier stopped.")

if __name__ == "__main__":
    run_with_logs()

