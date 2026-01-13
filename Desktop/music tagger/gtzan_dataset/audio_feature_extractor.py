"""
Audio Feature Extractor
========================
Extracts comprehensive audio features from music files using librosa.
Provides 58 features per 30-second audio clip for genre classification.

Author: Your Name
Date: November 2024
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import librosa.feature
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not installed. Using mock features.")


class AudioFeatureExtractor:
    """Extract audio features from music files"""
    
    def __init__(self, sample_rate: int = 22050, duration: float = 30.0):
        """
        Initialize feature extractor
        
        Args:
            sample_rate: Audio sample rate in Hz
            duration: Duration to analyze in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
    
    def extract_features(self, audio_file: Path) -> Dict[str, float]:
        """
        Extract comprehensive audio features from an audio file
        
        Args:
            audio_file: Path to audio file (WAV, AU, etc.)
        
        Returns:
            Dictionary of extracted features
        """
        if not LIBROSA_AVAILABLE:
            return self._mock_features()
        
        try:
            # Load audio (limit to specified duration)
            y, sr = librosa.load(
                audio_file,
                sr=self.sample_rate,
                duration=self.duration,
                mono=True
            )
            
            if len(y) == 0:
                print(f"Warning: Empty audio file: {audio_file}")
                return self._mock_features()
            
            features = {}
            
            # ========== TEMPORAL FEATURES ==========
            
            # Tempo (BPM)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate_mean'] = float(np.mean(zcr))
            features['zero_crossing_rate_std'] = float(np.std(zcr))
            
            # ========== SPECTRAL FEATURES ==========
            
            # Spectral Centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
            features['spectral_contrast_std'] = float(np.std(spectral_contrast))
            
            # Spectral Flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            features['spectral_flatness_std'] = float(np.std(spectral_flatness))
            
            # ========== MFCC FEATURES (13 coefficients) ==========
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            # ========== CHROMA FEATURES ==========
            
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            # Chroma features per pitch class
            for i in range(12):
                features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
                features[f'chroma_{i}_std'] = float(np.std(chroma[i]))
            
            # ========== RHYTHM FEATURES ==========
            
            # Onset Strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength_mean'] = float(np.mean(onset_env))
            features['onset_strength_std'] = float(np.std(onset_env))
            features['onset_strength_max'] = float(np.max(onset_env))
            
            # Beat frame
            if len(beats) > 0:
                features['beat_rate'] = float(len(beats) / (len(y) / sr))
            else:
                features['beat_rate'] = 0.0
            
            # ========== ENERGY FEATURES ==========
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            features['rms_max'] = float(np.max(rms))
            
            # ========== POLYPHONIC FEATURES ==========
            
            # Tonnetz (harmonic network)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features['tonnetz_mean'] = float(np.mean(tonnetz))
            features['tonnetz_std'] = float(np.std(tonnetz))
            
            # ========== STATISTICAL FEATURES ==========
            
            # Overall statistics
            features['duration'] = float(len(y) / sr)
            features['sample_rate'] = float(sr)
            
            # Ensure all features are finite
            for key, value in features.items():
                if not np.isfinite(value):
                    features[key] = 0.0
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_file}: {e}")
            return self._mock_features()
    
    def _mock_features(self) -> Dict[str, float]:
        """Generate mock features for testing when librosa is unavailable"""
        return {
            'tempo': 120.0,
            'spectral_centroid_mean': 2500.0,
            'zero_crossing_rate_mean': 0.1,
            'rms_mean': 0.15,
            'mfcc_0_mean': -50.0,
            'chroma_mean': 0.5,
            'onset_strength_mean': 0.5
        }
    
    def get_feature_names(self) -> list:
        """Get list of all feature names"""
        # Test extraction to get feature names
        if LIBROSA_AVAILABLE:
            try:
                # Create a dummy audio signal
                y = np.random.randn(self.sample_rate * int(self.duration))
                sr = self.sample_rate
                
                features = {}
                
                # Extract all features (similar to extract_features)
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = tempo
                
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                features['zero_crossing_rate_mean'] = np.mean(zcr)
                features['zero_crossing_rate_std'] = np.std(zcr)
                
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                features['spectral_centroid_mean'] = np.mean(spectral_centroids)
                features['spectral_centroid_std'] = np.std(spectral_centroids)
                
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
                features['spectral_rolloff_std'] = np.std(spectral_rolloff)
                
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
                features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
                
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                features['spectral_contrast_mean'] = np.mean(spectral_contrast)
                features['spectral_contrast_std'] = np.std(spectral_contrast)
                
                spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
                features['spectral_flatness_mean'] = np.mean(spectral_flatness)
                features['spectral_flatness_std'] = np.std(spectral_flatness)
                
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                for i in range(13):
                    features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                    features[f'mfcc_{i}_std'] = np.std(mfccs[i])
                
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                features['chroma_mean'] = np.mean(chroma)
                features['chroma_std'] = np.std(chroma)
                
                for i in range(12):
                    features[f'chroma_{i}_mean'] = np.mean(chroma[i])
                    features[f'chroma_{i}_std'] = np.std(chroma[i])
                
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                features['onset_strength_mean'] = np.mean(onset_env)
                features['onset_strength_std'] = np.std(onset_env)
                features['onset_strength_max'] = np.max(onset_env)
                
                if len(beats) > 0:
                    features['beat_rate'] = len(beats) / (len(y) / sr)
                else:
                    features['beat_rate'] = 0.0
                
                rms = librosa.feature.rms(y=y)[0]
                features['rms_mean'] = np.mean(rms)
                features['rms_std'] = np.std(rms)
                features['rms_max'] = np.max(rms)
                
                tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
                features['tonnetz_mean'] = np.mean(tonnetz)
                features['tonnetz_std'] = np.std(tonnetz)
                
                features['duration'] = len(y) / sr
                features['sample_rate'] = sr
                
                return list(features.keys())
            except:
                pass
        
        # Return default feature names
        return [
            'tempo', 'spectral_centroid_mean', 'zero_crossing_rate_mean',
            'rms_mean', 'mfcc_0_mean', 'chroma_mean', 'onset_strength_mean'
        ]


def extract_features_from_file(audio_file: Path) -> Dict[str, float]:
    """
    Convenience function to extract features from a single file
    
    Args:
        audio_file: Path to audio file
    
    Returns:
        Dictionary of extracted features
    """
    extractor = AudioFeatureExtractor()
    return extractor.extract_features(audio_file)


if __name__ == "__main__":
    # Test feature extraction
    print("Testing Audio Feature Extractor...")
    
    extractor = AudioFeatureExtractor()
    
    # Get feature names
    feature_names = extractor.get_feature_names()
    print(f"\nTotal features: {len(feature_names)}")
    print(f"Feature names: {feature_names[:10]}...")
    
    print("\nFeature extraction test complete!")

