"""
Test AWS Bedrock Setup
======================
Quick test script to verify AWS Bedrock configuration and model access
before running the full classification pipeline.

Author: Your Name
Date: November 2024
"""

import sys
from pathlib import Path

def test_aws_credentials():
    """Test AWS credentials"""
    print("Testing AWS credentials...")
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"[OK] AWS credentials configured")
        print(f"  Account: {identity.get('Account', 'N/A')}")
        print(f"  User: {identity.get('Arn', 'N/A')}")
        return True
    except Exception as e:
        print(f"[ERROR] AWS credentials not configured: {e}")
        return False

def test_bedrock_access():
    """Test Bedrock access"""
    print("\nTesting Bedrock access...")
    try:
        from bedrock_client import BedrockClient
        client = BedrockClient()
        
        if client.test_connection():
            print("[OK] Bedrock access successful")
            return True
        else:
            print("[ERROR] Bedrock access failed")
            return False
    except Exception as e:
        print(f"[ERROR] Error testing Bedrock: {e}")
        return False

def test_model_access():
    """Test model access"""
    print("\nTesting model access...")
    try:
        from bedrock_client import BedrockClient
        client = BedrockClient()
        
        # Test Mixtral access
        mixtral_id = client.MODELS["mixtral"]
        if client.check_model_access(mixtral_id):
            print(f"[OK] Mixtral model accessible: {mixtral_id}")
        else:
            print(f"[ERROR] Mixtral model not accessible: {mixtral_id}")
            print("  Please enable model access in Bedrock console")
        
        # Test Claude access
        claude_id = client.MODELS["claude_sonnet"]
        if client.check_model_access(claude_id):
            print(f"[OK] Claude Sonnet model accessible: {claude_id}")
        else:
            print(f"[ERROR] Claude Sonnet model not accessible: {claude_id}")
            print("  Please enable model access in Bedrock console")
        
        return True
    except Exception as e:
        print(f"✗ Error testing model access: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction"""
    print("\nTesting feature extraction...")
    try:
        from audio_feature_extractor import AudioFeatureExtractor
        extractor = AudioFeatureExtractor()
        
        # Get feature names
        feature_names = extractor.get_feature_names()
        print(f"[OK] Feature extraction available")
        print(f"  Total features: {len(feature_names)}")
        print(f"  Sample features: {feature_names[:5]}")
        return True
    except Exception as e:
        print(f"[ERROR] Error testing feature extraction: {e}")
        return False

def test_feature_description():
    """Test feature description"""
    print("\nTesting feature description...")
    try:
        from feature_descriptor import FeatureDescriptor
        descriptor = FeatureDescriptor()
        
        # Test with sample features
        sample_features = {
            'tempo': 120.0,
            'spectral_centroid_mean': 2500.0,
            'zero_crossing_rate_mean': 0.1,
            'rms_mean': 0.15
        }
        
        description = descriptor.create_feature_description(sample_features)
        print(f"[OK] Feature description available")
        print(f"  Sample description: {description[:100]}...")
        return True
    except Exception as e:
        print(f"[ERROR] Error testing feature description: {e}")
        return False

def test_prompt_generation():
    """Test prompt generation"""
    print("\nTesting prompt generation...")
    try:
        from prompt_generator import HybridPromptGenerator
        generator = HybridPromptGenerator()
        
        # Test with sample features
        sample_features = {
            'tempo': 120.0,
            'spectral_centroid_mean': 2500.0,
            'zero_crossing_rate_mean': 0.1,
            'rms_mean': 0.15,
            'onset_strength_mean': 0.5
        }
        sample_desc = "Moderate tempo at 120 BPM. Bright and clear tonal quality."
        
        prompt = generator.create_hybrid_prompt(sample_features, sample_desc)
        print(f"[OK] Prompt generation available")
        print(f"  Prompt length: {len(prompt)} characters")
        return True
    except Exception as e:
        print(f"[ERROR] Error testing prompt generation: {e}")
        return False

def test_simple_inference():
    """Test simple inference"""
    print("\nTesting simple inference...")
    try:
        from bedrock_client import BedrockClient
        from prompt_generator import HybridPromptGenerator
        from feature_descriptor import FeatureDescriptor
        
        # Create components
        client = BedrockClient()
        generator = HybridPromptGenerator()
        descriptor = FeatureDescriptor()
        
        # Test features
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
        
        # Create description
        feature_desc = descriptor.create_feature_description(sample_features)
        
        # Create prompt (simplified for testing)
        prompt = generator.create_simple_prompt(sample_features, feature_desc)
        
        # Test inference
        print("  Invoking Mixtral model...")
        result = client.invoke_mixtral(prompt, temperature=0.3, max_tokens=500)
        
        print(f"[OK] Inference successful")
        print(f"  Predicted genre: {result.get('genre', 'unknown')}")
        print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
        return True
    except Exception as e:
        print(f"[ERROR] Error testing inference: {e}")
        print("  This is normal if models are not enabled or credentials are not configured")
        return False

def test_dataset_access():
    """Test dataset access"""
    print("\nTesting dataset access...")
    try:
        import kagglehub
        dataset_path = kagglehub.dataset_download(
            "andradaolteanu/gtzan-dataset-music-genre-classification"
        )
        dataset_path = Path(dataset_path)
        
        if dataset_path.exists():
            print(f"[OK] Dataset accessible")
            print(f"  Path: {dataset_path}")
            
            # Check for genres directory
            genres_path = dataset_path / "Data" / "genres_original"
            if not genres_path.exists():
                genres_path = dataset_path / "genres_original"
            
            if genres_path.exists():
                genres = [d.name for d in genres_path.iterdir() if d.is_dir()]
                print(f"  Genres found: {len(genres)}")
                print(f"  Genre list: {', '.join(genres[:5])}...")
            return True
        else:
            print(f"[ERROR] Dataset not found")
            return False
    except Exception as e:
        print(f"[ERROR] Error accessing dataset: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print(" AWS Bedrock Setup Test ".center(60))
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results["AWS Credentials"] = test_aws_credentials()
    results["Bedrock Access"] = test_bedrock_access()
    results["Model Access"] = test_model_access()
    results["Feature Extraction"] = test_feature_extraction()
    results["Feature Description"] = test_feature_description()
    results["Prompt Generation"] = test_prompt_generation()
    results["Simple Inference"] = test_simple_inference()
    results["Dataset Access"] = test_dataset_access()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "[OK]" if passed else "[ERROR]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] All tests passed! System is ready to use.")
        print("\nYou can now run:")
        print("  python bedrock_music_classifier.py")
    else:
        print("[WARNING] Some tests failed. Please fix the issues above.")
        print("\nCommon issues:")
        print("  1. AWS credentials not configured → Run: aws configure")
        print("  2. Model access denied → Enable models in Bedrock console")
        print("  3. Dataset not found → Run: python download_dataset.py")
        print("  4. Missing dependencies → Run: pip install -r requirements_bedrock.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()

