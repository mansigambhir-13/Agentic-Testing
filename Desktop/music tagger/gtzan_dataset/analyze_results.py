"""Analyze and interpret classification results in natural language"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_results():
    results_file = Path("bedrock_results/tagged_songs.json")
    
    if not results_file.exists():
        print("Results file not found!")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    correct = sum(1 for s in data if s['is_correct'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Genre-wise statistics
    genre_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'predictions': defaultdict(int)})
    
    for song in data:
        actual = song['actual_genre']
        predicted = song['predicted_genre']
        genre_stats[actual]['total'] += 1
        genre_stats[actual]['predictions'][predicted] += 1
        if song['is_correct']:
            genre_stats[actual]['correct'] += 1
    
    # Average confidence
    avg_confidence = sum(s['confidence'] for s in data) / total if total > 0 else 0
    
    # Latest 10 songs
    latest = data[-10:]
    
    print("=" * 70)
    print(" MUSIC GENRE CLASSIFICATION RESULTS - NATURAL LANGUAGE INTERPRETATION")
    print("=" * 70)
    print()
    
    print(f"OVERALL PROGRESS:")
    print(f"  - Total songs processed: {total} out of 1000 ({total/10:.1f}%)")
    print(f"  - Correct classifications: {correct}")
    print(f"  - Overall accuracy: {accuracy:.2f}%")
    print(f"  - Average confidence: {avg_confidence:.2f}")
    print()
    
    print("GENRE-WISE PERFORMANCE:")
    print("-" * 70)
    for genre in sorted(genre_stats.keys()):
        stats = genre_stats[genre]
        genre_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        top_mistake = max(stats['predictions'].items(), key=lambda x: x[1]) if stats['predictions'] else (None, 0)
        
        print(f"\n{genre.upper()}:")
        print(f"  - Processed: {stats['total']} songs")
        print(f"  - Correct: {stats['correct']} ({genre_acc:.1f}% accuracy)")
        if top_mistake[0] and top_mistake[0] != genre:
            print(f"  - Most common mistake: Confused with '{top_mistake[0]}' ({top_mistake[1]} times)")
        print(f"  - Prediction distribution:")
        for pred_genre, count in sorted(stats['predictions'].items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    * {pred_genre}: {count} songs")
    
    print()
    print("LATEST 10 CLASSIFICATIONS:")
    print("-" * 70)
    for i, song in enumerate(latest, 1):
        status = "CORRECT" if song['is_correct'] else "INCORRECT"
        print(f"{i}. {song['file']}")
        print(f"   Actual: {song['actual_genre']} | Predicted: {song['predicted_genre']} | Confidence: {song['confidence']:.2f}")
        print(f"   Status: {status}")
        print()
    
    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print()
    
    # Natural language interpretation
    if accuracy < 20:
        print("The model is currently showing very low accuracy (<20%). This suggests:")
        print("  - The LLM may be struggling to distinguish between genres based on audio features alone")
        print("  - There might be systematic biases (e.g., predicting certain genres more often)")
        print("  - The feature descriptions may need refinement")
    elif accuracy < 40:
        print("The model shows low accuracy (20-40%). Observations:")
        print("  - The classification is better than random (10% baseline) but needs improvement")
        print("  - Some genres may be easier to identify than others")
        print("  - Consider analyzing which genres are most confused")
    elif accuracy < 60:
        print("The model shows moderate accuracy (40-60%). This indicates:")
        print("  - The LLM can identify some genre patterns from audio features")
        print("  - There's room for improvement through better prompting or feature engineering")
    else:
        print("The model shows good accuracy (>60%). This suggests:")
        print("  - The LLM is successfully learning genre patterns from audio features")
        print("  - The feature-to-text conversion is working well")
    
    print()
    print("KEY OBSERVATIONS:")
    print("-" * 70)
    
    # Find most confused genres
    confusion_pairs = []
    for genre, stats in genre_stats.items():
        for pred_genre, count in stats['predictions'].items():
            if pred_genre != genre and count > 0:
                confusion_pairs.append((genre, pred_genre, count))
    
    if confusion_pairs:
        top_confusion = max(confusion_pairs, key=lambda x: x[2])
        print(f"1. Most common confusion: '{top_confusion[0]}' songs are often misclassified as '{top_confusion[1]}'")
        print(f"   (This happened {top_confusion[2]} times)")
    
    # Find best performing genre
    best_genre = max(genre_stats.items(), key=lambda x: (x[1]['correct'] / x[1]['total']) if x[1]['total'] > 0 else 0)
    best_acc = (best_genre[1]['correct'] / best_genre[1]['total'] * 100) if best_genre[1]['total'] > 0 else 0
    print(f"\n2. Best performing genre: '{best_genre[0]}' with {best_acc:.1f}% accuracy")
    
    # Find worst performing genre
    worst_genre = min(genre_stats.items(), key=lambda x: (x[1]['correct'] / x[1]['total']) if x[1]['total'] > 0 else 1)
    worst_acc = (worst_genre[1]['correct'] / worst_genre[1]['total'] * 100) if worst_genre[1]['total'] > 0 else 0
    print(f"3. Most challenging genre: '{worst_genre[0]}' with {worst_acc:.1f}% accuracy")
    
    print()
    print("RECOMMENDATIONS:")
    print("-" * 70)
    if accuracy < 30:
        print("1. Consider refining the prompt to be more specific about genre characteristics")
        print("2. Add more few-shot examples for better context")
        print("3. Review if the audio features are being described accurately")
        print("4. Check if the model is consistently predicting certain genres (bias)")
    else:
        print("1. Continue monitoring as more songs are processed")
        print("2. Analyze patterns in misclassifications to improve prompts")
        print("3. Consider genre-specific feature emphasis in prompts")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    analyze_results()

