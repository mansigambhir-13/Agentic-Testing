# Single-sample classification runner for AWS Bedrock pipeline
# Usage: python run_single_sample.py <genre> <filename>

import json
import sys
from pathlib import Path
from datetime import datetime

from audio_feature_extractor import AudioFeatureExtractor
from feature_descriptor import FeatureDescriptor
from prompt_generator import HybridPromptGenerator
from bedrock_client import BedrockClient


def classify_single_track(audio_path: Path, actual_genre: str, region: str = "us-east-1", model_name: str = "mixtral") -> dict:
    """Classify a single audio track using the existing Bedrock pipeline."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    feature_extractor = AudioFeatureExtractor()
    feature_descriptor = FeatureDescriptor()
    prompt_generator = HybridPromptGenerator()
    bedrock_client = BedrockClient(region=region)

    features = feature_extractor.extract_features(audio_path)
    if not features:
        raise RuntimeError("Feature extraction failed for the provided audio file.")

    feature_desc = feature_descriptor.create_feature_description(features)
    prompt = prompt_generator.create_hybrid_prompt(features, feature_desc)
    prediction = bedrock_client.invoke_model(
        model_name=model_name,
        prompt=prompt,
        temperature=0.3,
        max_tokens=2000
    )

    predicted_genre = prediction.get("genre", "unknown")
    confidence = prediction.get("confidence", 0.0)
    is_correct = predicted_genre.lower() == actual_genre.lower()

    result = {
        "file": audio_path.name,
        "path": str(audio_path),
        "timestamp": datetime.now().isoformat(),
        "actual_genre": actual_genre,
        "predicted_genre": predicted_genre,
        "confidence": confidence,
        "is_correct": is_correct,
        "reasoning": prediction.get("reasoning", ""),
        "full_response": prediction,
        "feature_description": feature_desc,
    }

    return result


def main():
    dataset_root = Path(
        "C:/Users/LENOVO/.cache/kagglehub/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1"
    )
    genres_dir = dataset_root / "Data" / "genres_original"

    if len(sys.argv) == 3:
        genre = sys.argv[1].lower()
        filename = sys.argv[2]
        audio_file = genres_dir / genre / filename
    else:
        # Default sample: blues.00000.wav
        genre = "blues"
        audio_file = genres_dir / genre / "blues.00000.wav"
        print("No arguments supplied; defaulting to blues/blues.00000.wav")

    try:
        result = classify_single_track(audio_file, genre)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
