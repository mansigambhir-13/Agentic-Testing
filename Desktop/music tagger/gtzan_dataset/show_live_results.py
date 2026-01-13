"""
Show Live Classification Results
=================================
Display real-time results as songs are being processed.
"""

import json
import time
from pathlib import Path
from collections import defaultdict

def show_live_results():
    """Show live results"""
    results_dir = Path("bedrock_results")
    results_file = results_dir / "tagged_songs.json"
    
    print("=" * 60)
    print(" Live Classification Results ".center(60))
    print("=" * 60)
    
    last_count = 0
    
    try:
        while True:
            if results_file.exists():
                with open(results_file, 'r') as f:
                    songs = json.load(f)
                
                current_count = len(songs)
                
                if current_count > last_count:
                    # New songs processed
                    new_songs = songs[last_count:]
                    last_count = current_count
                    
                    # Calculate statistics
                    correct = sum(1 for s in songs if s.get('is_correct', False))
                    accuracy = (correct / current_count * 100) if current_count > 0 else 0
                    avg_confidence = sum(s.get('confidence', 0) for s in songs) / current_count if current_count > 0 else 0
                    
                    # Show latest predictions
                    print(f"\n[{time.strftime('%H:%M:%S')}] Processed {current_count}/1000 songs")
                    print(f"  Overall Accuracy: {accuracy:.2f}% ({correct}/{current_count})")
                    print(f"  Average Confidence: {avg_confidence:.2f}")
                    
                    # Show latest 5 predictions
                    print(f"\n  Latest Predictions:")
                    for song in new_songs[-5:]:
                        status = "[OK]" if song.get('is_correct', False) else "[ERROR]"
                        print(f"    {status} {song['file'][:25]:<25} -> {song['predicted_genre']:>8} "
                              f"(actual: {song['actual_genre']:>8}, conf: {song.get('confidence', 0):.2f})")
                    
                    # Genre-wise stats
                    genre_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
                    for song in songs:
                        genre = song.get('actual_genre', 'unknown')
                        genre_stats[genre]['total'] += 1
                        if song.get('is_correct', False):
                            genre_stats[genre]['correct'] += 1
                    
                    if current_count % 50 == 0:
                        print(f"\n  Genre-wise Performance:")
                        for genre in sorted(genre_stats.keys()):
                            stats = genre_stats[genre]
                            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                            print(f"    {genre:>10}: {acc:>5.1f}% ({stats['correct']:>3}/{stats['total']:>3})")
                    
                    print("-" * 60)
                else:
                    # No new songs, just show status
                    if current_count > 0:
                        correct = sum(1 for s in songs if s.get('is_correct', False))
                        accuracy = (correct / current_count * 100) if current_count > 0 else 0
                        print(f"\r[{time.strftime('%H:%M:%S')}] Waiting... {current_count}/1000 processed "
                              f"(Accuracy: {accuracy:.2f}%)", end='', flush=True)
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] Waiting for results file...", end='', flush=True)
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if results_file.exists():
            with open(results_file, 'r') as f:
                songs = json.load(f)
            print(f"\nFinal Results: {len(songs)} songs processed")
            if len(songs) > 0:
                correct = sum(1 for s in songs if s.get('is_correct', False))
                accuracy = (correct / len(songs) * 100)
                print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{len(songs)})")

if __name__ == "__main__":
    show_live_results()

