"""
Advanced Prompt Generator
==========================
Creates sophisticated prompts for music genre classification using
a hybrid strategy combining few-shot learning, chain-of-thought reasoning,
and multi-expert analysis for optimal LLM performance.

Author: Your Name
Date: November 2024
"""

from typing import Dict, List
import json


class HybridPromptGenerator:
    """Generate advanced prompts for music genre classification"""
    
    # Genre characteristics for context
    GENRE_CHARACTERISTICS = {
        "blues": {
            "tempo_range": "60-100 BPM",
            "rhythm": "12-bar structure, shuffle, swing",
            "instruments": "guitar, harmonica, piano",
            "vocals": "emotional, call-response, blue notes",
            "key_features": "guitar bends, slow to moderate tempo, expressive"
        },
        "classical": {
            "tempo_range": "varies widely (60-180 BPM)",
            "rhythm": "orchestral, complex time signatures",
            "instruments": "orchestral (strings, woodwinds, brass)",
            "vocals": "opera, choral, or instrumental",
            "key_features": "complex harmonies, wide dynamic range, no drums"
        },
        "country": {
            "tempo_range": "90-120 BPM",
            "rhythm": "steady, 4/4 time, twang",
            "instruments": "acoustic guitar, fiddle, steel guitar",
            "vocals": "storytelling, twang, clear",
            "key_features": "acoustic instruments, moderate tempo, narrative lyrics"
        },
        "disco": {
            "tempo_range": "110-130 BPM",
            "rhythm": "four-on-floor, syncopated bassline",
            "instruments": "drums, bass, strings, horns",
            "vocals": "dance-oriented, energetic",
            "key_features": "four-on-floor beat, 120-130 BPM, hi-hat patterns"
        },
        "hiphop": {
            "tempo_range": "70-100 BPM",
            "rhythm": "heavy bass, drum machines, sampling",
            "instruments": "samples, drum machines, synthesizers",
            "vocals": "rap, rhythmic, spoken word",
            "key_features": "heavy bass, 85-95 BPM, rhythmic vocals, syncopation"
        },
        "jazz": {
            "tempo_range": "varies (60-200 BPM)",
            "rhythm": "swing, syncopation, improvisation",
            "instruments": "piano, saxophone, trumpet, bass",
            "vocals": "scat, smooth, or instrumental",
            "key_features": "swing rhythm, complex chords, improvisation, warm tone"
        },
        "metal": {
            "tempo_range": "140-200 BPM",
            "rhythm": "fast, aggressive, double bass drums",
            "instruments": "distorted guitars, heavy drums, bass",
            "vocals": "aggressive, screamed, or growled",
            "key_features": "distorted guitars, fast tempo, high energy, bright spectrum"
        },
        "pop": {
            "tempo_range": "100-130 BPM",
            "rhythm": "steady, danceable, verse-chorus",
            "instruments": "synthesizers, drums, guitars",
            "vocals": "catchy, polished, auto-tuned",
            "key_features": "catchy melodies, verse-chorus structure, polished production"
        },
        "reggae": {
            "tempo_range": "60-90 BPM",
            "rhythm": "offbeat emphasis, emphasis on 3rd beat",
            "instruments": "bass, drums, guitar (skank), organ",
            "vocals": "laid-back, smooth",
            "key_features": "offbeat rhythm, bass-heavy, 60-90 BPM, relaxed tempo"
        },
        "rock": {
            "tempo_range": "110-140 BPM",
            "rhythm": "driving beat, 4/4 time, power chords",
            "instruments": "electric guitars, drums, bass",
            "vocals": "energetic, verse-chorus",
            "key_features": "electric guitars, drums, 110-140 BPM, power chords, guitar solos"
        }
    }
    
    # Few-shot examples
    FEW_SHOT_EXAMPLES = [
        {
            "genre": "metal",
            "features": {
                "tempo": 165,
                "spectral_centroid_mean": 3800,
                "zero_crossing_rate_mean": 0.18,
                "rms_mean": 0.28,
                "onset_strength_mean": 0.75
            },
            "description": "Very fast tempo (intense) at 165 BPM. Very bright and treble-focused tonal quality. Highly percussive and noisy texture. Very loud and intense energy level. Strong rhythmic accents with aggressive beat emphasis.",
            "reasoning": "The fast tempo (165 BPM), high energy (0.28 RMS), bright spectrum (3800 Hz), and highly percussive texture (0.18 ZCR) are all strong indicators of metal music. The aggressive rhythmic accents and intense energy level further confirm this classification."
        },
        {
            "genre": "classical",
            "features": {
                "tempo": 72,
                "spectral_centroid_mean": 2100,
                "zero_crossing_rate_mean": 0.04,
                "rms_mean": 0.08,
                "onset_strength_mean": 0.25
            },
            "description": "Slow tempo (relaxed, laid-back) at 72 BPM. Warm and balanced tonal quality. Smooth and sustained sounds with minimal percussive elements. Moderate energy level with dynamic range. Subtle rhythmic accents with gentle transitions.",
            "reasoning": "The slow tempo (72 BPM), smooth texture (0.04 ZCR), moderate energy (0.08 RMS), and warm tonal quality (2100 Hz) are characteristic of classical music. The subtle rhythmic accents and wide dynamic range further support this classification."
        },
        {
            "genre": "blues",
            "features": {
                "tempo": 88,
                "spectral_centroid_mean": 2200,
                "zero_crossing_rate_mean": 0.09,
                "rms_mean": 0.16,
                "onset_strength_mean": 0.48
            },
            "description": "Slow-moderate tempo (laid-back groove) at 88 BPM. Warm, mid-focused tonal quality. Moderate percussiveness with a shuffle-like feel. Strong but not harsh energy. Clearly articulated downbeats with swing accents.",
            "reasoning": "The 88 BPM tempo, warm spectrum (2200 Hz), and moderate texture (0.09 ZCR) align with blues. Energy is present but not aggressive (0.16 RMS) and the rhythmic feel is swung rather than straight, signaling blues instead of rock or reggae."
        },
        {
            "genre": "country",
            "features": {
                "tempo": 108,
                "spectral_centroid_mean": 2400,
                "zero_crossing_rate_mean": 0.11,
                "rms_mean": 0.18,
                "onset_strength_mean": 0.52
            },
            "description": "Moderate tempo (steady two-step) at 108 BPM. Balanced, slightly bright tonal quality. Percussive texture but still organic. Moderate, consistent energy. Clear rhythmic accents without heavy syncopation.",
            "reasoning": "The 108 BPM tempo, balanced spectrum (2400 Hz), and organic texture (0.11 ZCR) are typical of country. Energy (0.18 RMS) is lively but not aggressive, matching acoustic instrumentation rather than dense rock or hiphop production."
        },
        {
            "genre": "hiphop",
            "features": {
                "tempo": 92,
                "spectral_centroid_mean": 1400,
                "zero_crossing_rate_mean": 0.12,
                "rms_mean": 0.22,
                "onset_strength_mean": 0.55
            },
            "description": "Moderate tempo (walking pace, steady) at 92 BPM. Dark and bass-heavy tonal quality. Percussive and rhythmic texture. High energy and powerful sound. Moderate rhythmic accents with clear beat emphasis.",
            "reasoning": "The moderate tempo (92 BPM), bass-heavy spectrum (1400 Hz), percussive texture (0.12 ZCR), and high energy (0.22 RMS) are typical of hip-hop. The rhythmic beat emphasis and bass-forward mix are key indicators."
        },
        {
            "genre": "rock",
            "features": {
                "tempo": 132,
                "spectral_centroid_mean": 3000,
                "zero_crossing_rate_mean": 0.15,
                "rms_mean": 0.24,
                "onset_strength_mean": 0.68
            },
            "description": "Fast tempo (energetic, driving) at 132 BPM. Bright, guitar-focused spectrum. Pronounced percussive hits with snare accents. Loud, saturated energy. Strong rhythmic drive with backbeat emphasis.",
            "reasoning": "The 132 BPM tempo, bright spectrum (3000 Hz), high percussiveness (0.15 ZCR), and elevated energy (0.24 RMS) strongly indicate rock. Consistent backbeat hits and electric guitar brightness differentiate it from disco or metal."
        },
        {
            "genre": "jazz",
            "features": {
                "tempo": 120,
                "spectral_centroid_mean": 2500,
                "zero_crossing_rate_mean": 0.08,
                "rms_mean": 0.12,
                "onset_strength_mean": 0.45
            },
            "description": "Fast tempo (energetic, driving) at 120 BPM. Bright and clear tonal quality. Moderate percussiveness with a mix of smooth and rhythmic elements. Moderate energy level with dynamic range. Moderate rhythmic accents with clear beat emphasis.",
            "reasoning": "The moderate-fast tempo (120 BPM), warm to bright tonal quality (2500 Hz), mixed texture (0.08 ZCR), and dynamic energy (0.12 RMS) are characteristic of jazz. The swing rhythm and complex harmonic content further support this classification."
        },
        {
            "genre": "reggae",
            "features": {
                "tempo": 76,
                "spectral_centroid_mean": 1600,
                "zero_crossing_rate_mean": 0.09,
                "rms_mean": 0.15,
                "onset_strength_mean": 0.35
            },
            "description": "Slow tempo (relaxed, laid-back) at 76 BPM. Dark and bass-heavy tonal quality. Moderate percussiveness with a mix of smooth and rhythmic elements. High energy and powerful sound. Subtle rhythmic accents with gentle transitions.",
            "reasoning": "The slow tempo (76 BPM), bass-heavy spectrum (1600 Hz), offbeat rhythm, and relaxed energy are characteristic of reggae. The bass-forward mix and laid-back tempo are key indicators."
        }
    ]
    
    DECISION_RULES = [
        "If tempo < 90 BPM AND zero crossing rate < 0.08 AND RMS < 0.12 -> strongly consider classical, jazz ballad, or acoustic blues. Avoid hiphop/reggae labels unless percussion is clearly strong.",
        "If zero crossing rate < 0.06 and onset strength < 0.35 -> the track likely lacks drum transients; favor classical or ambient genres over hiphop, rock, or reggae.",
        "If RMS > 0.22 and spectral centroid > 2800 Hz -> the mix is loud and bright; candidate genres are rock, metal, or disco rather than blues or country.",
        "If tempo is between 95-115 BPM with warm spectrum (1800-2600 Hz) and moderate texture (0.09-0.13 ZCR) -> lean toward country or blues unless energy is extremely high.",
        "If the feature description explicitly mentions 'offbeat emphasis' or 'syncopated bassline' with tempo 60-95 BPM -> reggae is more plausible than rock or country.",
        "Before finalizing hiphop, confirm there is bass-heavy spectrum (<1800 Hz), strong percussion (ZCR >= 0.11), and pronounced rhythmic accents (onset >= 0.5)."
    ]
    
    def create_hybrid_prompt(self, features: Dict[str, float], 
                            feature_desc: str) -> str:
        """
        Create hybrid prompt combining few-shot, chain-of-thought, and multi-expert
        
        Args:
            features: Dictionary of audio features
            feature_desc: Natural language description of features
        
        Returns:
            Complete prompt string
        """
        prompt = f"""You are an expert music analyst with deep knowledge of music genres and audio characteristics. 
Your task is to classify a 30-second audio clip into one of these 10 genres:
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

=== SYSTEM INSTRUCTIONS ===

1. Analyze the audio features systematically
2. Use chain-of-thought reasoning
3. Consider multiple expert perspectives
4. Provide confidence-calibrated predictions
5. Return results in JSON format only

=== GENRE CHARACTERISTICS REFERENCE ===

{self._format_genre_characteristics()}

=== FEW-SHOT EXAMPLES ===

{self._format_few_shot_examples()}

=== DECISION RULES TO APPLY ===

{self._format_decision_rules()}

=== CURRENT TRACK TO CLASSIFY ===

Audio Analysis:
{feature_desc}

Detailed Feature Measurements:
- Tempo: {features.get('tempo', 0):.1f} BPM
- Spectral Centroid (brightness): {features.get('spectral_centroid_mean', 0):.1f} Hz
- Zero Crossing Rate (texture): {features.get('zero_crossing_rate_mean', 0):.4f}
- RMS Energy (loudness): {features.get('rms_mean', 0):.4f}
- Onset Strength (rhythm): {features.get('onset_strength_mean', 0):.4f}
- Spectral Bandwidth: {features.get('spectral_bandwidth_mean', 0):.1f} Hz
- MFCC-0 (timbre): {features.get('mfcc_0_mean', 0):.2f}
- Chroma (harmonic): {features.get('chroma_mean', 0):.3f}

=== CHAIN-OF-THOUGHT ANALYSIS ===

Please analyze this track step-by-step:

Step 1: Tempo Analysis
- Current tempo: {features.get('tempo', 0):.1f} BPM
- Which genres typically have this tempo range?
- Eliminate genres that don't match this tempo

Step 2: Spectral Analysis
- Spectral centroid: {features.get('spectral_centroid_mean', 0):.1f} Hz
- What does this brightness level indicate?
- Which genres match this tonal quality?

Step 3: Texture Analysis
- Zero crossing rate: {features.get('zero_crossing_rate_mean', 0):.4f}
- How percussive vs smooth is this track?
- Which genres have this texture?

Step 4: Energy Analysis
- RMS energy: {features.get('rms_mean', 0):.4f}
- What is the overall loudness/energy level?
- Which genres match this energy profile?

Step 5: Rhythm Analysis
- Onset strength: {features.get('onset_strength_mean', 0):.4f}
- How strong are the rhythmic accents?
- Which genres have this rhythmic character?

Step 6: Synthesis
- Combine all observations from Steps 1-5
- Which genre best matches all characteristics?
- What is your confidence level?

=== MULTI-EXPERT PERSPECTIVE ===

Consider three expert opinions:

Expert 1 - Rhythm Specialist:
Focus on tempo ({features.get('tempo', 0):.1f} BPM) and rhythm patterns.
Which genre does this tempo suggest?

Expert 2 - Frequency Analyst:
Focus on spectral content (centroid: {features.get('spectral_centroid_mean', 0):.1f} Hz).
What does the frequency distribution indicate?

Expert 3 - Energy Expert:
Focus on dynamics (RMS: {features.get('rms_mean', 0):.4f}) and energy distribution.
What does the loudness profile suggest?

Consensus:
- Do all experts agree?
- If not, which expert's analysis is most relevant?
- What is the final classification?

=== CONFIDENCE CALIBRATION ===

Rate your confidence based on:
- High (0.9-1.0): All features strongly match one genre, no ambiguity
- Medium-High (0.7-0.9): Most features match, minor ambiguities
- Medium (0.5-0.7): Features match 2-3 genres equally, some ambiguity
- Low (0.3-0.5): Generic features, many genres possible
- Very Low (0.0-0.3): Contradictory indicators or unusual combination

=== OUTPUT FORMAT ===

Respond ONLY with valid JSON in this exact format:

{{
  "genre": "your_classification",
  "confidence": 0.0 to 1.0,
  "reasoning": "detailed explanation of why you chose this genre",
  "step1_tempo": "tempo analysis and genre candidates",
  "step2_spectral": "spectral analysis and genre candidates",
  "step3_texture": "texture analysis and genre candidates",
  "step4_energy": "energy analysis and genre candidates",
  "step5_rhythm": "rhythm analysis and genre candidates",
  "step6_synthesis": "how all factors combine",
  "expert1_rhythm": "rhythm expert's opinion",
  "expert2_frequency": "frequency analyst's opinion",
  "expert3_energy": "energy expert's opinion",
  "consensus": "final consensus from all experts",
  "confidence_explanation": "why this confidence level",
  "alternative_genres": ["second_choice", "third_choice"],
  "key_indicators": ["indicator1", "indicator2", "indicator3"],
  "eliminated_genres": ["genre1", "genre2"],
  "strongest_matches": ["genre1", "genre2"]
}}

IMPORTANT:
- Return ONLY the JSON object
- Do not include any additional text before or after
- Ensure all JSON fields are properly formatted
- Genre must be exactly one of: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
"""
        return prompt
    
    def _format_genre_characteristics(self) -> str:
        """Format genre characteristics for prompt"""
        lines = []
        for genre, chars in self.GENRE_CHARACTERISTICS.items():
            lines.append(f"\n{genre.upper()}:")
            lines.append(f"  Tempo: {chars['tempo_range']}")
            lines.append(f"  Rhythm: {chars['rhythm']}")
            lines.append(f"  Instruments: {chars['instruments']}")
            lines.append(f"  Key Features: {chars['key_features']}")
        return "\n".join(lines)
    
    def _format_decision_rules(self) -> str:
        """Format decision rules for prompt"""
        lines = []
        for idx, rule in enumerate(self.DECISION_RULES, 1):
            lines.append(f"{idx}. {rule}")
        return "\n".join(lines)
    
    def _format_few_shot_examples(self) -> str:
        """Format few-shot examples for prompt"""
        lines = []
        for i, example in enumerate(self.FEW_SHOT_EXAMPLES, 1):
            lines.append(f"\nExample {i} - {example['genre'].upper()}:")
            lines.append(f"  Features: Tempo={example['features']['tempo']} BPM, "
                        f"Spectral={example['features']['spectral_centroid_mean']} Hz, "
                        f"ZCR={example['features']['zero_crossing_rate_mean']:.2f}, "
                        f"Energy={example['features']['rms_mean']:.2f}")
            lines.append(f"  Description: {example['description']}")
            lines.append(f"  Reasoning: {example['reasoning']}")
            lines.append(f"  → Genre: {example['genre']} (high confidence)")
        return "\n".join(lines)
    
    def create_simple_prompt(self, features: Dict[str, float], 
                            feature_desc: str) -> str:
        """
        Create simpler prompt for faster/cheaper models
        
        Args:
            features: Dictionary of audio features
            feature_desc: Natural language description of features
        
        Returns:
            Simplified prompt string
        """
        prompt = f"""Classify this music track into one genre: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

Audio Features:
{feature_desc}

Key Measurements:
- Tempo: {features.get('tempo', 0):.1f} BPM
- Brightness: {features.get('spectral_centroid_mean', 0):.1f} Hz
- Texture: {features.get('zero_crossing_rate_mean', 0):.4f}
- Energy: {features.get('rms_mean', 0):.4f}

Respond in JSON format:
{{
  "genre": "your_classification",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}"""
        return prompt


if __name__ == "__main__":
    # Test prompt generation
    print("Testing Hybrid Prompt Generator...")
    
    generator = HybridPromptGenerator()
    
    # Test with sample features
    sample_features = {
        'tempo': 165.0,
        'spectral_centroid_mean': 3800.0,
        'zero_crossing_rate_mean': 0.18,
        'rms_mean': 0.28,
        'onset_strength_mean': 0.75,
        'spectral_bandwidth_mean': 3500.0,
        'mfcc_0_mean': -30.0,
        'chroma_mean': 0.4
    }
    
    sample_desc = "Very fast tempo (intense) at 165 BPM. Very bright and treble-focused tonal quality. Highly percussive and noisy texture. Very loud and intense energy level."
    
    prompt = generator.create_hybrid_prompt(sample_features, sample_desc)
    
    print(f"\nGenerated prompt length: {len(prompt)} characters")
    print(f"\nPrompt preview (first 500 chars):")
    print(prompt[:500])
    
    print("\nPrompt generation test complete!")

