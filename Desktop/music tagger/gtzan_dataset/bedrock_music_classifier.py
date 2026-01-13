"""
AWS Bedrock Music Genre Classifier
===================================
Main script for classifying music genres using AWS Bedrock models.
Processes GTZAN dataset and tags each song with genre predictions.

Author: Your Name
Date: November 2024
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from audio_feature_extractor import AudioFeatureExtractor
from feature_descriptor import FeatureDescriptor
from prompt_generator import HybridPromptGenerator
from bedrock_client import BedrockClient


class BedrockMusicClassifier:
    """Main classifier for music genre classification using AWS Bedrock"""
    
    def __init__(self, region: str = "us-east-1", 
                 model_name: str = "mixtral",
                 samples_per_genre: int = 10):
        """
        Initialize classifier
        
        Args:
            region: AWS region for Bedrock
            model_name: Model to use (mixtral, claude_sonnet, claude_haiku)
            samples_per_genre: Number of samples to process per genre
        """
        self.region = region
        self.model_name = model_name
        self.samples_per_genre = samples_per_genre
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor()
        self.feature_descriptor = FeatureDescriptor()
        self.prompt_generator = HybridPromptGenerator()
        self.bedrock_client = BedrockClient(region=region)
        
        # Results storage
        self.results = []
        self.tagged_songs = []
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "total_correct": 0,
            "total_errors": 0,
            "total_time": 0,
            "avg_confidence": 0
        }
    
    def process_single_song(self, audio_file: Path, actual_genre: str) -> Dict:
        """
        Process a single song and get genre prediction
        
        Args:
            audio_file: Path to audio file
            actual_genre: Actual genre label
        
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio_file)
            
            if not features:
                print(f"Warning: Could not extract features from {audio_file.name}")
                return {
                    "file": audio_file.name,
                    "actual_genre": actual_genre,
                    "predicted_genre": "error",
                    "confidence": 0.0,
                    "error": "Feature extraction failed"
                }
            
            # Create feature description
            feature_desc = self.feature_descriptor.create_feature_description(features)
            
            # Generate prompt
            prompt = self.prompt_generator.create_hybrid_prompt(features, feature_desc)
            
            # Get prediction from Bedrock
            prediction = self.bedrock_client.invoke_model(
                model_name=self.model_name,
                prompt=prompt,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Process time
            process_time = time.time() - start_time
            
            # Extract results
            predicted_genre = prediction.get("genre", "unknown")
            confidence = prediction.get("confidence", 0.0)
            reasoning = prediction.get("reasoning", "")
            
            # Check if correct
            is_correct = predicted_genre.lower() == actual_genre.lower()
            
            # Update statistics
            self.stats["total_processed"] += 1
            if is_correct:
                self.stats["total_correct"] += 1
            if predicted_genre == "error":
                self.stats["total_errors"] += 1
            self.stats["total_time"] += process_time
            self.stats["avg_confidence"] = (
                (self.stats["avg_confidence"] * (self.stats["total_processed"] - 1) + confidence) 
                / self.stats["total_processed"]
            )
            
            # Create result dictionary
            result = {
                "file": audio_file.name,
                "actual_genre": actual_genre,
                "predicted_genre": predicted_genre,
                "confidence": confidence,
                "is_correct": is_correct,
                "reasoning": reasoning,
                "features": features,
                "feature_description": feature_desc,
                "process_time": process_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add additional prediction details if available
            if "key_indicators" in prediction:
                result["key_indicators"] = prediction["key_indicators"]
            if "alternative_genres" in prediction:
                result["alternative_genres"] = prediction["alternative_genres"]
            if "step1_tempo" in prediction:
                result["chain_of_thought"] = {
                    "step1_tempo": prediction.get("step1_tempo", ""),
                    "step2_spectral": prediction.get("step2_spectral", ""),
                    "step3_texture": prediction.get("step3_texture", ""),
                    "step4_energy": prediction.get("step4_energy", ""),
                    "step5_rhythm": prediction.get("step5_rhythm", ""),
                    "step6_synthesis": prediction.get("step6_synthesis", "")
                }
            
            return result
            
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            self.stats["total_errors"] += 1
            return {
                "file": audio_file.name,
                "actual_genre": actual_genre,
                "predicted_genre": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def process_dataset(self, dataset_path: Path) -> List[Dict]:
        """
        Process GTZAN dataset
        
        Args:
            dataset_path: Path to GTZAN dataset
        
        Returns:
            List of prediction results
        """
        print("\n" + "=" * 60)
        print("Processing GTZAN Dataset")
        print("=" * 60)
        
        # Find genres directory
        genres_path = dataset_path / "Data" / "genres_original"
        if not genres_path.exists():
            genres_path = dataset_path / "genres_original"
        
        if not genres_path.exists():
            print("Error: Could not find genres_original directory")
            return []
        
        # Get all genres
        genres = sorted([d.name for d in genres_path.iterdir() if d.is_dir()])
        print(f"\nFound {len(genres)} genres: {', '.join(genres)}")
        
        # Process each genre
        all_results = []
        
        for genre in genres:
            genre_path = genres_path / genre
            audio_files = sorted(list(genre_path.glob("*.wav")) + 
                               list(genre_path.glob("*.au")))
            
            # Limit to samples_per_genre
            audio_files = audio_files[:self.samples_per_genre]
            
            print(f"\nProcessing {genre} ({len(audio_files)} files)...")
            print(f"  Progress: 0/{len(audio_files)}", end='', flush=True)
            
            for idx, audio_file in enumerate(audio_files, 1):
                result = self.process_single_song(audio_file, genre)
                all_results.append(result)
                
                # Print result with progress
                pred_genre = result.get("predicted_genre", "unknown")
                confidence = result.get("confidence", 0.0)
                is_correct = result.get("is_correct", False)
                status = "[OK]" if is_correct else "[ERROR]"
                
                print(f"\r  Progress: {idx}/{len(audio_files)} | {status} {audio_file.name[:25]:<25} -> {pred_genre:>8} "
                      f"(actual: {genre:>8}, conf: {confidence:.2f})", end='', flush=True)
                
                # Save intermediate results every 10 songs
                if idx % 10 == 0:
                    self.results = all_results
                    self.save_results(Path("bedrock_results"))
                    print(f" | Saved", end='', flush=True)
                
                # Rate limiting (avoid hitting API limits)
                time.sleep(0.5)  # 0.5 second delay between requests
            
            print()  # New line after genre complete
        
        self.results = all_results
        return all_results
    
    def calculate_accuracy(self) -> Dict:
        """
        Calculate accuracy metrics
        
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.results:
            return {}
        
        # Overall accuracy
        total = len([r for r in self.results if r.get("predicted_genre") != "error"])
        correct = len([r for r in self.results if r.get("is_correct", False)])
        accuracy = correct / total if total > 0 else 0
        
        # Genre-wise accuracy
        genre_stats = {}
        for result in self.results:
            if result.get("predicted_genre") == "error":
                continue
            
            actual = result["actual_genre"]
            if actual not in genre_stats:
                genre_stats[actual] = {"correct": 0, "total": 0, "confidences": []}
            
            genre_stats[actual]["total"] += 1
            if result.get("is_correct", False):
                genre_stats[actual]["correct"] += 1
            genre_stats[actual]["confidences"].append(result.get("confidence", 0))
        
        # Calculate per-genre accuracy
        for genre, stats in genre_stats.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            stats["avg_confidence"] = np.mean(stats["confidences"]) if stats["confidences"] else 0
        
        return {
            "overall_accuracy": accuracy,
            "total_correct": correct,
            "total_processed": total,
            "genre_stats": genre_stats,
            "avg_confidence": self.stats["avg_confidence"],
            "avg_process_time": self.stats["total_time"] / total if total > 0 else 0
        }
    
    def create_confusion_matrix(self) -> pd.DataFrame:
        """
        Create confusion matrix
        
        Returns:
            DataFrame with confusion matrix
        """
        if not self.results:
            return pd.DataFrame()
        
        # Get all genres
        genres = sorted(set([
            r["actual_genre"] for r in self.results
        ] + [
            r["predicted_genre"] for r in self.results 
            if r.get("predicted_genre") != "error"
        ]))
        
        # Create confusion matrix
        confusion = {actual: {pred: 0 for pred in genres} for actual in genres}
        
        for result in self.results:
            if result.get("predicted_genre") == "error":
                continue
            
            actual = result["actual_genre"]
            predicted = result["predicted_genre"]
            
            if actual in confusion and predicted in confusion[actual]:
                confusion[actual][predicted] += 1
        
        # Convert to DataFrame
        df = pd.DataFrame(confusion).T
        df = df.fillna(0).astype(int)
        
        return df
    
    def visualize_results(self, output_dir: Path):
        """
        Create visualizations of results (DISABLED - not generating visualizations)
        
        Args:
            output_dir: Directory to save visualizations
        """
        # Visualization generation disabled per user request
        return
        if not self.results:
            print("No results to visualize")
            return
        
        # Calculate accuracy
        accuracy_metrics = self.calculate_accuracy()
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Overall accuracy
        ax = axes[0, 0]
        overall_acc = accuracy_metrics.get("overall_accuracy", 0)
        ax.barh(["Overall"], [overall_acc], color='skyblue')
        ax.set_xlabel('Accuracy')
        ax.set_xlim([0, 1])
        ax.set_title(f'Overall Accuracy: {overall_acc:.2%}')
        ax.text(overall_acc / 2, 0, f'{overall_acc:.2%}', 
               ha='center', va='center', fontsize=14, fontweight='bold')
        
        # 2. Genre-wise accuracy
        ax = axes[0, 1]
        genre_stats = accuracy_metrics.get("genre_stats", {})
        if genre_stats:
            genres = sorted(genre_stats.keys())
            accuracies = [genre_stats[g]["accuracy"] for g in genres]
            
            bars = ax.bar(genres, accuracies, color='lightcoral')
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0, 1])
            ax.set_title('Genre-wise Accuracy')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{acc:.2%}', ha='center', va='bottom')
        
        # 3. Confusion matrix
        ax = axes[1, 0]
        confusion_df = self.create_confusion_matrix()
        if not confusion_df.empty:
            sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Genre')
            ax.set_ylabel('Actual Genre')
            ax.set_title('Confusion Matrix')
        
        # 4. Confidence distribution
        ax = axes[1, 1]
        confidences = [r.get("confidence", 0) for r in self.results 
                      if r.get("predicted_genre") != "error"]
        if confidences:
            ax.hist(confidences, bins=20, color='lightgreen', edgecolor='black')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Frequency')
            ax.set_title('Confidence Distribution')
            ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(confidences):.2f}')
            ax.legend()
        
        plt.suptitle(f'Bedrock Music Genre Classification Results ({self.model_name})', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'classification_results.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved visualization to {output_dir / 'classification_results.png'}")
    
    def save_results(self, output_dir: Path):
        """
        Save results to files
        
        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tagged songs
        tagged_songs = []
        for result in self.results:
            tagged_songs.append({
                "file": result["file"],
                "actual_genre": result["actual_genre"],
                "predicted_genre": result.get("predicted_genre", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "is_correct": result.get("is_correct", False),
                "reasoning": result.get("reasoning", "")[:500],  # Limit length
                "key_indicators": result.get("key_indicators", []),
                "alternative_genres": result.get("alternative_genres", [])
            })
        
        with open(output_dir / 'tagged_songs.json', 'w') as f:
            json.dump(tagged_songs, f, indent=2)
        
        # Save accuracy metrics
        accuracy_metrics = self.calculate_accuracy()
        with open(output_dir / 'accuracy_metrics.json', 'w') as f:
            json.dump(accuracy_metrics, f, indent=2, default=str)
        
        # Save confusion matrix
        confusion_df = self.create_confusion_matrix()
        if not confusion_df.empty:
            confusion_df.to_csv(output_dir / 'confusion_matrix.csv')
        
        # Create detailed report
        report = self.create_report(accuracy_metrics)
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\n[OK] Results saved to {output_dir}/")
        print(f"  - tagged_songs.json ({len(tagged_songs)} songs)")
        print(f"  - accuracy_metrics.json")
        print(f"  - confusion_matrix.csv")
        print(f"  - classification_report.txt")
    
    def create_report(self, accuracy_metrics: Dict) -> str:
        """
        Create detailed classification report
        
        Args:
            accuracy_metrics: Dictionary with accuracy metrics
        
        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("AWS BEDROCK MUSIC GENRE CLASSIFICATION REPORT")
        report.append("=" * 60)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {self.model_name}")
        report.append(f"Region: {self.region}")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total Processed: {self.stats['total_processed']}")
        report.append(f"Total Correct: {self.stats['total_correct']}")
        report.append(f"Total Errors: {self.stats['total_errors']}")
        report.append(f"Overall Accuracy: {accuracy_metrics.get('overall_accuracy', 0):.2%}")
        report.append(f"Average Confidence: {accuracy_metrics.get('avg_confidence', 0):.2f}")
        report.append(f"Average Process Time: {accuracy_metrics.get('avg_process_time', 0):.2f} seconds")
        report.append("")
        
        # Genre-wise statistics
        report.append("GENRE-WISE PERFORMANCE:")
        report.append("-" * 40)
        genre_stats = accuracy_metrics.get("genre_stats", {})
        for genre in sorted(genre_stats.keys()):
            stats = genre_stats[genre]
            report.append(f"\n{genre.upper()}:")
            report.append(f"  Accuracy: {stats['accuracy']:.2%}")
            report.append(f"  Correct: {stats['correct']}/{stats['total']}")
            report.append(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
        
        report.append("")
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main execution function"""
    print("=" * 60)
    print(" AWS Bedrock Music Genre Classifier ".center(60))
    print("=" * 60)
    
    # Configuration
    REGION = "us-east-1"  # Change if needed
    MODEL_NAME = "mixtral"  # Options: mixtral, claude_sonnet, claude_haiku
    SAMPLES_PER_GENRE = 100  # Number of songs to process per genre (100 = all songs)
    
    # Check AWS credentials
    try:
        import boto3
        boto3.client('sts').get_caller_identity()
        print("[OK] AWS credentials configured")
    except Exception as e:
        print(f"[ERROR] AWS credentials not configured: {e}")
        print("\nPlease configure AWS CLI:")
        print("  aws configure")
        print("\nOr set environment variables:")
        print("  export AWS_ACCESS_KEY_ID=your_key")
        print("  export AWS_SECRET_ACCESS_KEY=your_secret")
        print("  export AWS_DEFAULT_REGION=us-east-1")
        return
    
    # Initialize classifier
    print(f"\nInitializing classifier...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Region: {REGION}")
    print(f"  Samples per genre: {SAMPLES_PER_GENRE}")
    
    classifier = BedrockMusicClassifier(
        region=REGION,
        model_name=MODEL_NAME,
        samples_per_genre=SAMPLES_PER_GENRE
    )
    
    # Test Bedrock connection
    print("\nTesting Bedrock connection...")
    if not classifier.bedrock_client.test_connection():
        print("[ERROR] Bedrock connection failed")
        print("\nPlease ensure:")
        print("1. You have Bedrock access in your AWS account")
        print("2. Required models are enabled in Bedrock console")
        print("3. You have permissions to invoke models")
        return
    
    print("[OK] Bedrock connection successful")
    
    # Load dataset
    try:
        import kagglehub
        print("\n" + "=" * 60)
        print("Loading GTZAN Dataset")
        print("=" * 60)
        
        dataset_path = kagglehub.dataset_download(
            "andradaolteanu/gtzan-dataset-music-genre-classification"
        )
        dataset_path = Path(dataset_path)
        print(f"[OK] Dataset loaded from: {dataset_path}")
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("Please ensure the dataset is downloaded")
        return
    
    # Process dataset
    print("\n" + "=" * 60)
    print("Processing Dataset")
    print("=" * 60)
    print(f"\nThis will process {SAMPLES_PER_GENRE} songs per genre")
    estimated_time = SAMPLES_PER_GENRE * 10 * 5 / 60
    print(f"Estimated time: ~{estimated_time:.1f} minutes")
    print("\nStarting processing...")
    
    results = classifier.process_dataset(dataset_path)
    
    # Calculate accuracy
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)
    
    accuracy_metrics = classifier.calculate_accuracy()
    
    print(f"\nOverall Accuracy: {accuracy_metrics.get('overall_accuracy', 0):.2%}")
    print(f"Total Correct: {accuracy_metrics.get('total_correct', 0)}/{accuracy_metrics.get('total_processed', 0)}")
    print(f"Average Confidence: {accuracy_metrics.get('avg_confidence', 0):.2f}")
    
    # Genre-wise performance
    print("\nGenre-wise Performance:")
    genre_stats = accuracy_metrics.get("genre_stats", {})
    for genre in sorted(genre_stats.keys()):
        stats = genre_stats[genre]
        print(f"  {genre}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save results (visualization disabled)
    output_dir = Path("bedrock_results")
    classifier.save_results(output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nKey Findings:")
    print(f"  • Overall Accuracy: {accuracy_metrics.get('overall_accuracy', 0):.2%}")
    print(f"  • Best Genre: {max(genre_stats.items(), key=lambda x: x[1]['accuracy'])[0] if genre_stats else 'N/A'}")
    print(f"  • Most Confident: {max([r for r in results if r.get('confidence', 0) > 0], key=lambda x: x.get('confidence', 0), default={}).get('predicted_genre', 'N/A')}")
    print(f"\nNext Steps:")
    print(f"  1. Review tagged_songs.json for all predictions")
    print(f"  2. Check classification_report.txt for detailed analysis")
    print(f"  3. Examine confusion_matrix.csv for genre confusion patterns")


if __name__ == "__main__":
    main()

