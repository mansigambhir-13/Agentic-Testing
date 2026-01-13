"""
View Live Classification Logs
==============================
View the classification logs in real-time as they're generated.
"""

import time
import os
from pathlib import Path

def view_live_logs():
    """View live logs from classification"""
    log_file = Path("classification_logs.txt")
    
    print("=" * 60)
    print(" Live Classification Logs Viewer ".center(60))
    print("=" * 60)
    print("\nWatching for logs... Press Ctrl+C to stop\n")
    
    if not log_file.exists():
        print(f"Waiting for log file: {log_file}")
        print("The classifier should be running and creating logs...")
    
    last_size = 0
    
    try:
        while True:
            if log_file.exists():
                current_size = log_file.stat().st_size
                
                if current_size > last_size:
                    # New content available
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        if new_content:
                            print(new_content, end='', flush=True)
                    last_size = current_size
                else:
                    # No new content, show status
                    print(f"\r[{time.strftime('%H:%M:%S')}] Waiting for new logs... (File size: {current_size} bytes)", 
                          end='', flush=True)
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] Waiting for log file to be created...", end='', flush=True)
            
            time.sleep(1)  # Check every second
            
    except KeyboardInterrupt:
        print("\n\nLog viewing stopped.")
        if log_file.exists():
            print(f"\nLog file location: {log_file.absolute()}")
            print(f"Total log size: {log_file.stat().st_size} bytes")

if __name__ == "__main__":
    view_live_logs()

