"""
Watch Classification Progress in Real-Time
===========================================
Shows live progress updates as songs are being classified.
Run this in a separate terminal to watch the progress.
"""

import json
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def watch_progress():
    """Watch classification progress in real-time"""
    results_file = Path("bedrock_results/tagged_songs.json")
    
    print("=" * 70)
    print(" REAL-TIME CLASSIFICATION PROGRESS ".center(70))
    print("=" * 70)
    print("\nWatching for results... Press Ctrl+C to stop\n")
    
    last_count = 0
    start_time = time.time()
    
    try:
        while True:
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        songs = json.load(f)
                    
                    current_count = len(songs)
                    
                    if current_count > last_count:
                        # New songs processed
                        elapsed = time.time() - start_time
                        rate = current_count / elapsed if elapsed > 0 else 0
                        remaining = 1000 - current_count
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta_minutes = eta_seconds / 60
                        
                        # Calculate statistics
                        correct = sum(1 for s in songs if s.get('is_correct', False))
                        accuracy = (correct / current_count * 100) if current_count > 0 else 0
                        avg_confidence = sum(s.get('confidence', 0) for s in songs) / current_count if current_count > 0 else 0
                        
                        # Genre statistics
                        genre_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
                        for song in songs:
                            genre = song.get('actual_genre', 'unknown')
                            genre_stats[genre]['total'] += 1
                            if song.get('is_correct', False):
                                genre_stats[genre]['correct'] += 1
                        
                        # Clear screen and show updated stats
                        print("\033[2J\033[H", end='')  # Clear screen
                        print("=" * 70)
                        print(" REAL-TIME CLASSIFICATION PROGRESS ".center(70))
                        print("=" * 70)
                        print(f"\nTime: {datetime.now().strftime('%H:%M:%S')}")
                        print(f"Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s")
                        print(f"Rate: {rate:.2f} songs/min")
                        print(f"ETA: {int(eta_minutes)}m {int(eta_seconds % 60)}s")
                        print("\n" + "-" * 70)
                        print(f"Progress: {current_count}/1000 songs ({current_count/10:.1f}%)")
                        print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{current_count})")
                        print(f"Average Confidence: {avg_confidence:.2f}")
                        print("\n" + "-" * 70)
                        print("Genre-wise Performance:")
                        print(f"{'Genre':<12} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Avg Conf':<10}")
                        print("-" * 70)
                        for genre in sorted(genre_stats.keys()):
                            stats = genre_stats[genre]
                            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                            avg_conf = sum(s.get('confidence', 0) for s in songs 
                                         if s.get('actual_genre') == genre) / stats['total'] if stats['total'] > 0 else 0
                            print(f"{genre:<12} {stats['correct']:<10} {stats['total']:<10} {acc:>6.1f}%    {avg_conf:>6.2f}")
                        
                        # Show latest predictions
                        print("\n" + "-" * 70)
                        print("Latest 5 Predictions:")
                        for song in songs[-5:]:
                            status = "[OK]" if song.get('is_correct', False) else "[ERROR]"
                            print(f"  {status} {song['file'][:25]:<25} -> {song['predicted_genre']:>8} "
                                  f"(actual: {song['actual_genre']:>8}, conf: {song.get('confidence', 0):.2f})")
                        
                        last_count = current_count
                        
                        if current_count >= 1000:
                            print("\n" + "=" * 70)
                            print(" CLASSIFICATION COMPLETE! ".center(70))
                            print("=" * 70)
                            break
                    else:
                        # No new songs, show waiting status
                        elapsed = time.time() - start_time
                        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Waiting... "
                              f"{current_count}/1000 processed (Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s)", 
                              end='', flush=True)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Error reading results: {e}", end='', flush=True)
            else:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Waiting for results file...", end='', flush=True)
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if results_file.exists():
            with open(results_file, 'r') as f:
                songs = json.load(f)
            print(f"\nFinal count: {len(songs)} songs processed")
            if len(songs) > 0:
                correct = sum(1 for s in songs if s.get('is_correct', False))
                accuracy = (correct / len(songs) * 100)
                print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{len(songs)})")

if __name__ == "__main__":
    watch_progress()

