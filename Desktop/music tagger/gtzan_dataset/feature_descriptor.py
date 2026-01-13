"""
Feature Descriptor
==================
Converts numerical audio features into natural language descriptions
that LLMs can understand for music genre classification.

Author: Your Name
Date: November 2024
"""

from typing import Dict, List


class FeatureDescriptor:
    """Convert audio features to natural language descriptions"""
    
    def __init__(self):
        """Initialize feature descriptor"""
        pass
    
    def describe_tempo(self, tempo: float) -> Dict[str, str]:
        """
        Describe tempo in natural language
        
        Args:
            tempo: Tempo in BPM
        
        Returns:
            Dictionary with description and category
        """
        if tempo < 60:
            return {
                "description": f"very slow tempo (ballad-like, contemplative) at {tempo:.0f} BPM",
                "category": "very_slow",
                "typical_genres": ["blues", "reggae", "classical"]
            }
        elif tempo < 90:
            return {
                "description": f"slow tempo (relaxed, laid-back) at {tempo:.0f} BPM",
                "category": "slow",
                "typical_genres": ["reggae", "blues", "hiphop", "jazz"]
            }
        elif tempo < 120:
            return {
                "description": f"moderate tempo (walking pace, steady) at {tempo:.0f} BPM",
                "category": "moderate",
                "typical_genres": ["hiphop", "pop", "country", "jazz"]
            }
        elif tempo < 140:
            return {
                "description": f"fast tempo (energetic, driving) at {tempo:.0f} BPM",
                "category": "fast",
                "typical_genres": ["rock", "pop", "disco", "country"]
            }
        else:
            return {
                "description": f"very fast tempo (intense, aggressive) at {tempo:.0f} BPM",
                "category": "very_fast",
                "typical_genres": ["metal", "punk", "drum_and_bass"]
            }
    
    def describe_spectral_centroid(self, centroid: float) -> Dict[str, str]:
        """
        Describe spectral centroid (brightness) in natural language
        
        Args:
            centroid: Spectral centroid in Hz
        
        Returns:
            Dictionary with description and category
        """
        if centroid < 1500:
            return {
                "description": f"dark and bass-heavy tonal quality (spectral centroid: {centroid:.0f} Hz)",
                "category": "dark",
                "typical_genres": ["hiphop", "reggae", "electronic"]
            }
        elif centroid < 2500:
            return {
                "description": f"warm and balanced tonal quality (spectral centroid: {centroid:.0f} Hz)",
                "category": "warm",
                "typical_genres": ["jazz", "blues", "country", "pop"]
            }
        elif centroid < 3500:
            return {
                "description": f"bright and clear tonal quality (spectral centroid: {centroid:.0f} Hz)",
                "category": "bright",
                "typical_genres": ["pop", "rock", "classical", "jazz"]
            }
        else:
            return {
                "description": f"very bright and treble-focused tonal quality (spectral centroid: {centroid:.0f} Hz)",
                "category": "very_bright",
                "typical_genres": ["metal", "punk", "aggressive_rock"]
            }
    
    def describe_zero_crossing_rate(self, zcr: float) -> Dict[str, str]:
        """
        Describe zero crossing rate (texture) in natural language
        
        Args:
            zcr: Zero crossing rate
        
        Returns:
            Dictionary with description and category
        """
        if zcr < 0.05:
            return {
                "description": f"smooth and sustained sounds with minimal percussive elements",
                "category": "smooth",
                "typical_genres": ["classical", "ambient", "jazz_ballad"]
            }
        elif zcr < 0.10:
            return {
                "description": f"moderate percussiveness with a mix of smooth and rhythmic elements",
                "category": "mixed",
                "typical_genres": ["jazz", "blues", "country", "pop"]
            }
        elif zcr < 0.15:
            return {
                "description": f"percussive and rhythmic texture",
                "category": "percussive",
                "typical_genres": ["rock", "pop", "hiphop", "disco"]
            }
        else:
            return {
                "description": f"highly percussive and noisy texture with strong rhythmic emphasis",
                "category": "highly_percussive",
                "typical_genres": ["metal", "punk", "aggressive_rock"]
            }
    
    def describe_rms_energy(self, rms: float) -> Dict[str, str]:
        """
        Describe RMS energy (loudness) in natural language
        
        Args:
            rms: RMS energy
        
        Returns:
            Dictionary with description and category
        """
        if rms < 0.05:
            return {
                "description": f"very quiet and subdued energy level",
                "category": "quiet",
                "typical_genres": ["classical", "ambient", "jazz_ballad"]
            }
        elif rms < 0.10:
            return {
                "description": f"moderate energy level with dynamic range",
                "category": "moderate",
                "typical_genres": ["jazz", "country", "blues", "pop"]
            }
        elif rms < 0.20:
            return {
                "description": f"high energy and powerful sound",
                "category": "high",
                "typical_genres": ["rock", "hiphop", "disco", "pop"]
            }
        else:
            return {
                "description": f"very loud and intense energy level",
                "category": "very_high",
                "typical_genres": ["metal", "punk", "heavy_rock"]
            }
    
    def describe_mfcc(self, mfcc_0: float) -> Dict[str, str]:
        """
        Describe MFCC (timbre) in natural language
        
        Args:
            mfcc_0: First MFCC coefficient
        
        Returns:
            Dictionary with description and category
        """
        if mfcc_0 < -100:
            return {
                "description": f"deep and resonant timbre characteristics",
                "category": "deep",
                "typical_genres": ["hiphop", "reggae", "electronic"]
            }
        elif mfcc_0 < -50:
            return {
                "description": f"rich and full timbre characteristics",
                "category": "rich",
                "typical_genres": ["jazz", "blues", "classical"]
            }
        else:
            return {
                "description": f"bright and thin timbre characteristics",
                "category": "bright",
                "typical_genres": ["pop", "rock", "metal"]
            }
    
    def describe_onset_strength(self, onset: float) -> Dict[str, str]:
        """
        Describe onset strength (rhythmic emphasis) in natural language
        
        Args:
            onset: Onset strength
        
        Returns:
            Dictionary with description and category
        """
        if onset < 0.3:
            return {
                "description": f"subtle rhythmic accents with gentle transitions",
                "category": "subtle",
                "typical_genres": ["classical", "ambient", "jazz_ballad"]
            }
        elif onset < 0.6:
            return {
                "description": f"moderate rhythmic accents with clear beat emphasis",
                "category": "moderate",
                "typical_genres": ["jazz", "blues", "country", "pop"]
            }
        else:
            return {
                "description": f"strong rhythmic accents with aggressive beat emphasis",
                "category": "strong",
                "typical_genres": ["rock", "metal", "hiphop", "disco"]
            }
    
    def create_feature_description(self, features: Dict[str, float]) -> str:
        """
        Create comprehensive natural language description of audio features
        
        Args:
            features: Dictionary of audio features
        
        Returns:
            Natural language description string
        """
        descriptions = []
        
        # Tempo description
        tempo = features.get('tempo', 0)
        if tempo > 0:
            tempo_desc = self.describe_tempo(tempo)
            descriptions.append(tempo_desc["description"])
        
        # Spectral centroid description
        centroid = features.get('spectral_centroid_mean', 0)
        if centroid > 0:
            centroid_desc = self.describe_spectral_centroid(centroid)
            descriptions.append(centroid_desc["description"])
        
        # Zero crossing rate description
        zcr = features.get('zero_crossing_rate_mean', 0)
        if zcr > 0:
            zcr_desc = self.describe_zero_crossing_rate(zcr)
            descriptions.append(zcr_desc["description"])
        
        # RMS energy description
        rms = features.get('rms_mean', 0)
        if rms > 0:
            rms_desc = self.describe_rms_energy(rms)
            descriptions.append(rms_desc["description"])
        
        # MFCC description
        mfcc_0 = features.get('mfcc_0_mean', 0)
        if mfcc_0 != 0:
            mfcc_desc = self.describe_mfcc(mfcc_0)
            descriptions.append(mfcc_desc["description"])
        
        # Onset strength description
        onset = features.get('onset_strength_mean', 0)
        if onset > 0:
            onset_desc = self.describe_onset_strength(onset)
            descriptions.append(onset_desc["description"])
        
        # Chroma description
        chroma = features.get('chroma_mean', 0)
        if chroma > 0:
            if chroma < 0.3:
                descriptions.append("sparse harmonic content with minimal chordal complexity")
            elif chroma < 0.6:
                descriptions.append("moderate harmonic content with clear tonal center")
            else:
                descriptions.append("rich harmonic content with complex chordal structure")
        
        # Spectral bandwidth description
        bandwidth = features.get('spectral_bandwidth_mean', 0)
        if bandwidth > 0:
            if bandwidth < 2000:
                descriptions.append("focused frequency spectrum with narrow bandwidth")
            elif bandwidth < 3500:
                descriptions.append("balanced frequency spectrum with moderate bandwidth")
            else:
                descriptions.append("wide frequency spectrum with broad bandwidth")
        
        # Combine descriptions
        if tempo > 0 and zcr > 0 and rms > 0:
            if tempo < 95 and zcr < 0.08 and rms < 0.15:
                descriptions.append("Overall profile is gentle and low-percussive, consistent with acoustic or orchestral styles (classical, jazz ballad, slow blues) and unlikely for hiphop, rock, or reggae.")
            elif 95 <= tempo <= 115 and 1800 <= centroid <= 2600 and 0.08 <= zcr <= 0.13 and 0.12 <= rms <= 0.20:
                descriptions.append("Steady mid-tempo groove with warm spectrum and moderate percussion aligns with country or blues more than reggae or hiphop.")
            elif tempo >= 125 and centroid >= 2800 and rms >= 0.20 and zcr >= 0.13:
                descriptions.append("Bright, high-energy mix with strong percussion favors rock or metal rather than blues or country.")
        if zcr < 0.06 and onset < 0.35:
            descriptions.append("Sparse transient activity suggests minimal drums, a hallmark of classical or ambient arrangements rather than beat-driven genres.")

        if descriptions:
            return ". ".join(descriptions) + "."
        else:
            return "Audio features extracted but unable to generate detailed description."
    
    def create_detailed_description(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Create detailed multi-level description with categorization
        
        Args:
            features: Dictionary of audio features
        
        Returns:
            Dictionary with detailed descriptions and categories
        """
        tempo = features.get('tempo', 0)
        centroid = features.get('spectral_centroid_mean', 0)
        zcr = features.get('zero_crossing_rate_mean', 0)
        rms = features.get('rms_mean', 0)
        mfcc_0 = features.get('mfcc_0_mean', 0)
        onset = features.get('onset_strength_mean', 0)
        
        # Get descriptions
        tempo_desc = self.describe_tempo(tempo) if tempo > 0 else None
        centroid_desc = self.describe_spectral_centroid(centroid) if centroid > 0 else None
        zcr_desc = self.describe_zero_crossing_rate(zcr) if zcr > 0 else None
        rms_desc = self.describe_rms_energy(rms) if rms > 0 else None
        mfcc_desc = self.describe_mfcc(mfcc_0) if mfcc_0 != 0 else None
        onset_desc = self.describe_onset_strength(onset) if onset > 0 else None
        
        # Combine typical genres from all descriptions
        typical_genres = set()
        for desc in [tempo_desc, centroid_desc, zcr_desc, rms_desc, mfcc_desc, onset_desc]:
            if desc and "typical_genres" in desc:
                typical_genres.update(desc["typical_genres"])
        
        return {
            "low_level": {
                "tempo": tempo_desc,
                "energy": rms_desc,
                "brightness": centroid_desc
            },
            "mid_level": {
                "texture": zcr_desc,
                "timbre": mfcc_desc,
                "rhythm": onset_desc
            },
            "high_level": {
                "overall_character": self.create_feature_description(features),
                "typical_genres": list(typical_genres)
            },
            "summary": self.create_feature_description(features)
        }


if __name__ == "__main__":
    # Test feature description
    print("Testing Feature Descriptor...")
    
    descriptor = FeatureDescriptor()
    
    # Test with sample features
    sample_features = {
        'tempo': 165.0,
        'spectral_centroid_mean': 3800.0,
        'zero_crossing_rate_mean': 0.18,
        'rms_mean': 0.28,
        'mfcc_0_mean': -30.0,
        'onset_strength_mean': 0.75,
        'chroma_mean': 0.4
    }
    
    description = descriptor.create_feature_description(sample_features)
    print(f"\nGenerated Description:")
    print(description)
    
    detailed = descriptor.create_detailed_description(sample_features)
    print(f"\nDetailed Description:")
    print(f"Typical genres: {detailed['high_level']['typical_genres']}")
    
    print("\nFeature description test complete!")

