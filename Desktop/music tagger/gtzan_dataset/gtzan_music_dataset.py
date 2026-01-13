"""
GTZAN Music Genre Classification Dataset
=========================================
Script to download and explore the GTZAN dataset from Kaggle
This dataset contains 1000 audio files (100 per genre) with extracted features and spectrograms


"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def install_dependencies():
    """Install required packages if not already installed"""
    required_packages = {
        'kagglehub': 'kagglehub',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'scikit-learn': 'scikit-learn',
        'tqdm': 'tqdm'
    }
    
    print("Checking dependencies...")
    for package, pip_name in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"{sys.executable} -m pip install {pip_name}")
    print()

def download_dataset() -> Path:
    """
    Download the GTZAN dataset using kagglehub
    
    Returns:
        Path to the downloaded dataset directory
    """
    try:
        import kagglehub
        
        print("=" * 60)
        print("Downloading GTZAN Music Genre Classification Dataset")
        print("=" * 60)
        
        # Download dataset
        dataset_path = kagglehub.dataset_download(
            "andradaolteanu/gtzan-dataset-music-genre-classification"
        )
        
        dataset_path = Path(dataset_path)
        print(f"\n✓ Dataset downloaded to: {dataset_path}")
        
        # List contents
        print("\nDataset structure:")
        for item in dataset_path.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
                # Show first few items in directories
                sub_items = list(item.iterdir())[:3]
                for sub_item in sub_items:
                    print(f"    └── {sub_item.name}")
                if len(list(item.iterdir())) > 3:
                    print(f"    └── ... ({len(list(item.iterdir()))} total items)")
            else:
                print(f"  📄 {item.name}")
        
        return dataset_path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease ensure you have:")
        print("1. kagglehub installed: pip install kagglehub")
        print("2. Kaggle API credentials configured")
        print("   - Go to kaggle.com -> Account -> Create API Token")
        print("   - Place kaggle.json in ~/.kaggle/")
        sys.exit(1)

def explore_csv_features(dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and explore the CSV feature files
    
    Args:
        dataset_path: Path to the dataset directory
    
    Returns:
        Tuple of (features_30_sec, features_3_sec) DataFrames
    """
    print("\n" + "=" * 60)
    print("Exploring Feature Files")
    print("=" * 60)
    
    # Load 30-second features
    features_30_path = dataset_path / "Data" / "features_30_sec.csv"
    features_3_path = dataset_path / "Data" / "features_3_sec.csv"
    
    if not features_30_path.exists():
        # Try alternative path structure
        features_30_path = dataset_path / "features_30_sec.csv"
        features_3_path = dataset_path / "features_3_sec.csv"
    
    print(f"\nLoading features from CSV files...")
    
    # Load DataFrames
    df_30 = pd.read_csv(features_30_path)
    df_3 = pd.read_csv(features_3_path)
    
    print(f"\n30-second features shape: {df_30.shape}")
    print(f"3-second features shape: {df_3.shape}")
    
    # Display basic info
    print("\n30-second features columns:")
    print(df_30.columns.tolist()[:10], "..." if len(df_30.columns) > 10 else "")
    
    print("\nFirst few rows of 30-second features:")
    print(df_30.head())
    
    print("\nGenre distribution (30-second):")
    if 'label' in df_30.columns:
        print(df_30['label'].value_counts())
    
    # Statistical summary of key features
    print("\nStatistical summary of audio features (30-second):")
    feature_cols = [col for col in df_30.columns if col not in ['filename', 'label', 'length']]
    print(df_30[feature_cols[:5]].describe())
    
    return df_30, df_3

def analyze_audio_features(df_30: pd.DataFrame, df_3: pd.DataFrame):
    """
    Perform basic analysis and visualization of audio features
    
    Args:
        df_30: DataFrame with 30-second features
        df_3: DataFrame with 3-second features
    """
    print("\n" + "=" * 60)
    print("Analyzing Audio Features")
    print("=" * 60)
    
    # Check for label column
    if 'label' not in df_30.columns:
        print("No 'label' column found in dataset")
        return
    
    # Create visualizations directory
    viz_dir = Path("gtzan_visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Genre distribution plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    genre_counts = df_30['label'].value_counts()
    genre_counts.plot(kind='bar')
    plt.title('Genre Distribution (30-second clips)')
    plt.xlabel('Genre')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    genre_counts_3 = df_3['label'].value_counts()
    genre_counts_3.plot(kind='bar', color='coral')
    plt.title('Genre Distribution (3-second clips)')
    plt.xlabel('Genre')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'genre_distribution.png', dpi=100, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: genre_distribution.png")
    
    # 2. Feature correlation heatmap (select key features)
    feature_cols = [col for col in df_30.columns if col not in ['filename', 'label', 'length']]
    
    # Select subset of features for visualization
    key_features = [col for col in feature_cols if any(
        keyword in col.lower() for keyword in ['mfcc1', 'spectral_centroid', 'zero_crossing_rate', 'tempo']
    )][:10]
    
    if key_features:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df_30[key_features].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_correlation.png', dpi=100, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: feature_correlation.png")
    
    # 3. Feature importance by genre (example with spectral features)
    spectral_features = [col for col in feature_cols if 'spectral' in col.lower()][:5]
    
    if spectral_features:
        fig, axes = plt.subplots(1, len(spectral_features), figsize=(15, 4))
        if len(spectral_features) == 1:
            axes = [axes]
        
        for idx, feature in enumerate(spectral_features):
            df_30.boxplot(column=feature, by='label', ax=axes[idx])
            axes[idx].set_title(feature.replace('_', ' ').title())
            axes[idx].set_xlabel('Genre')
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
        
        plt.suptitle('Spectral Features by Genre', y=1.02)
        plt.tight_layout()
        plt.savefig(viz_dir / 'spectral_features_by_genre.png', dpi=100, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: spectral_features_by_genre.png")

def explore_audio_files(dataset_path: Path):
    """
    Explore the actual audio files in the dataset
    
    Args:
        dataset_path: Path to the dataset directory
    """
    print("\n" + "=" * 60)
    print("Exploring Audio Files")
    print("=" * 60)
    
    # Find genres directory
    genres_path = dataset_path / "Data" / "genres_original"
    if not genres_path.exists():
        genres_path = dataset_path / "genres_original"
    
    if not genres_path.exists():
        print("Could not find genres_original directory")
        return
    
    # List all genres
    genres = [d.name for d in genres_path.iterdir() if d.is_dir()]
    print(f"\nFound {len(genres)} genres:")
    for genre in genres:
        genre_path = genres_path / genre
        audio_files = list(genre_path.glob("*.wav")) + list(genre_path.glob("*.au"))
        print(f"  • {genre}: {len(audio_files)} files")
    
    # Try to load and analyze one sample file if librosa is available
    try:
        import librosa
        import librosa.display
        
        # Get a sample file
        sample_genre = genres[0] if genres else None
        if sample_genre:
            sample_files = list((genres_path / sample_genre).glob("*.wav"))
            if not sample_files:
                sample_files = list((genres_path / sample_genre).glob("*.au"))
            
            if sample_files:
                sample_file = sample_files[0]
                print(f"\nAnalyzing sample file: {sample_file.name}")
                
                # Load audio
                y, sr = librosa.load(sample_file, sr=None)
                duration = len(y) / sr
                
                print(f"  Sample rate: {sr} Hz")
                print(f"  Duration: {duration:.2f} seconds")
                print(f"  Shape: {y.shape}")
                
                # Create visualizations
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # Waveform
                axes[0, 0].plot(np.linspace(0, duration, len(y)), y)
                axes[0, 0].set_title(f'Waveform - {sample_genre}')
                axes[0, 0].set_xlabel('Time (s)')
                axes[0, 0].set_ylabel('Amplitude')
                
                # Spectrogram
                D = librosa.stft(y)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = axes[0, 1].imshow(S_db, origin='lower', aspect='auto', cmap='viridis')
                axes[0, 1].set_title('Spectrogram')
                axes[0, 1].set_xlabel('Time')
                axes[0, 1].set_ylabel('Frequency')
                plt.colorbar(img, ax=axes[0, 1])
                
                # Mel-spectrogram
                mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
                mel_db = librosa.power_to_db(mel_spect, ref=np.max)
                img = axes[1, 0].imshow(mel_db, origin='lower', aspect='auto', cmap='magma')
                axes[1, 0].set_title('Mel-Spectrogram')
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Mel-Frequency')
                plt.colorbar(img, ax=axes[1, 0])
                
                # MFCCs
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                img = axes[1, 1].imshow(mfccs, origin='lower', aspect='auto', cmap='coolwarm')
                axes[1, 1].set_title('MFCCs')
                axes[1, 1].set_xlabel('Time')
                axes[1, 1].set_ylabel('MFCC Coefficients')
                plt.colorbar(img, ax=axes[1, 1])
                
                plt.suptitle(f'Audio Analysis: {sample_file.name}', y=1.02)
                plt.tight_layout()
                
                viz_dir = Path("gtzan_visualizations")
                viz_dir.mkdir(exist_ok=True)
                plt.savefig(viz_dir / 'audio_analysis_sample.png', dpi=100, bbox_inches='tight')
                plt.show()
                print(f"✓ Saved: audio_analysis_sample.png")
                
    except ImportError:
        print("\nNote: Install librosa for audio file analysis: pip install librosa")
    except Exception as e:
        print(f"\nError analyzing audio files: {e}")

def build_simple_classifier(df_30: pd.DataFrame):
    """
    Build a simple machine learning classifier for genre classification
    
    Args:
        df_30: DataFrame with 30-second features
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n" + "=" * 60)
        print("Building Simple Genre Classifier")
        print("=" * 60)
        
        if 'label' not in df_30.columns:
            print("No 'label' column found in dataset")
            return
        
        # Prepare features and labels
        feature_cols = [col for col in df_30.columns if col not in ['filename', 'label', 'length']]
        X = df_30[feature_cols].values
        y = df_30['label'].values
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Classes: {le.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        print("\nTraining Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = rf_model.score(X_train_scaled, y_train)
        test_score = rf_model.score(X_test_scaled, y_test)
        
        print(f"\nTrain Accuracy: {train_score:.3f}")
        print(f"Test Accuracy: {test_score:.3f}")
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Confusion Matrix - Random Forest Classifier')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        viz_dir = Path("gtzan_visualizations")
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / 'confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: confusion_matrix.png")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance['importance'].values)
        plt.yticks(range(len(feature_importance)), feature_importance['feature'].values)
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_importance.png', dpi=100, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: feature_importance.png")
        
    except ImportError:
        print("\nNote: Install scikit-learn for machine learning: pip install scikit-learn")
    except Exception as e:
        print(f"\nError building classifier: {e}")

def main():
    """Main execution function"""
    print("╔" + "=" * 58 + "╗")
    print("║" + " GTZAN Music Genre Classification Dataset Explorer ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Check and install dependencies
    install_dependencies()
    
    # Download dataset
    dataset_path = download_dataset()
    
    # Explore CSV features
    df_30, df_3 = explore_csv_features(dataset_path)
    
    # Analyze audio features
    analyze_audio_features(df_30, df_3)
    
    # Explore audio files
    explore_audio_files(dataset_path)
    
    # Build simple classifier
    build_simple_classifier(df_30)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nGenerated files saved to: ./gtzan_visualizations/")
    print("\nNext steps:")
    print("1. Explore the spectrograms in 'images_original' folder")
    print("2. Try deep learning models (CNN) on the spectrogram images")
    print("3. Experiment with different audio features for classification")
    print("4. Try data augmentation techniques for better model performance")
    print("\nFor more advanced analysis, consider:")
    print("  • Using pre-trained audio models (e.g., VGGish, YAMNet)")
    print("  • Implementing data augmentation (pitch shift, time stretch)")
    print("  • Creating ensemble models combining different features")
    print("  • Exploring transfer learning approaches")

if __name__ == "__main__":
    main()

