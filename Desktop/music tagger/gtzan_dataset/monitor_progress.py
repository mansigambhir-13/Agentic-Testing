"""
Monitor Classification Progress
==============================
Script to monitor the progress of the music genre classification in real-time.
"""

import json
import time
from pathlib import Path

def monitor_progress():
    """Monitor classification progress"""
    results_dir = Path("bedrock_results")
    
    print("=" * 60)
    print("Monitoring Classification Progress")
    print("=" * 60)
    print("\nPress Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            if (results_dir / "tagged_songs.json").exists():
                with open(results_dir / "tagged_songs.json", 'r') as f:
                    songs = json.load(f)
                
                total = len(songs)
                
                if total > 0:
                    # Calculate statistics
                    correct = sum(1 for s in songs if s.get('is_correct', False))
                    accuracy = (correct / total * 100) if total > 0 else 0
                    avg_confidence = sum(s.get('confidence', 0) for s in songs) / total if total > 0 else 0
                    
                    # Count by genre
                    genres = {}
                    for song in songs:
                        actual = song.get('actual_genre', 'unknown')
                        if actual not in genres:
                            genres[actual] = {'total': 0, 'correct': 0}
                        genres[actual]['total'] += 1
                        if song.get('is_correct', False):
                            genres[actual]['correct'] += 1
                    
                    # Display progress
                    print(f"\rProgress: {total}/1000 songs processed | "
                          f"Accuracy: {accuracy:.1f}% ({correct}/{total}) | "
                          f"Avg Confidence: {avg_confidence:.2f}", end='', flush=True)
                    
                    # Show genre breakdown every 50 songs
                    if total % 50 == 0:
                        print(f"\n\nGenre Progress:")
                        for genre in sorted(genres.keys()):
                            stats = genres[genre]
                            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                            print(f"  {genre}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
                        print()
                else:
                    print(f"\rWaiting for results... ({total} songs processed)", end='', flush=True)
            else:
                print(f"\rWaiting for results file...", end='', flush=True)
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if (results_dir / "tagged_songs.json").exists():
            with open(results_dir / "tagged_songs.json", 'r') as f:
                songs = json.load(f)
            print(f"\nFinal count: {len(songs)} songs processed")

if __name__ == "__main__":
    monitor_progress()

