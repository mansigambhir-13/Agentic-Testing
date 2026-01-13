# 🎵 Music Tagger - Advanced Music Genre Classification System

A comprehensive, production-ready system for automatic music genre classification using multiple approaches: traditional machine learning (CNN), and cutting-edge Large Language Models (AWS Bedrock). This project provides state-of-the-art tools for analyzing and classifying music genres from audio files.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AWS](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results & Performance](#results--performance)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)

## 🎯 Overview

Music Tagger is a sophisticated music genre classification system that leverages multiple approaches to achieve high-accuracy genre prediction:

1. **Traditional ML/CNN Approach**: Uses Convolutional Neural Networks on spectrogram images for deep learning-based classification
2. **LLM-Based Approach**: Utilizes AWS Bedrock (Mixtral, Claude) with advanced prompting strategies to classify genres using natural language descriptions of audio features

The system processes the GTZAN dataset (1000 songs across 10 genres) and can be extended to classify any audio file.

### Supported Genres

- 🎸 **Blues** - Slow to moderate tempo, emotional vocals, guitar-driven
- 🎼 **Classical** - Orchestral, complex harmonies, wide dynamic range
- 🤠 **Country** - Acoustic instruments, storytelling lyrics, twang
- 💃 **Disco** - Four-on-floor beat, dance-oriented, energetic
- 🎤 **Hip-hop** - Heavy bass, rhythmic vocals, sampling
- 🎷 **Jazz** - Swing rhythm, improvisation, complex chords
- 🤘 **Metal** - Fast tempo, distorted guitars, aggressive
- 🎵 **Pop** - Catchy melodies, polished production, verse-chorus
- 🎹 **Reggae** - Offbeat rhythm, bass-heavy, relaxed tempo
- 🎸 **Rock** - Electric guitars, driving beat, power chords

## ✨ Features

### Core Capabilities

- **Multi-Approach Classification**: Choose between CNN-based or LLM-based classification
- **Comprehensive Feature Extraction**: Extracts 58 audio features including tempo, spectral features, MFCCs, chroma, and rhythm features
- **Advanced Prompting**: Hybrid strategy combining few-shot learning, chain-of-thought reasoning, and multi-expert analysis
- **Production-Ready**: Comprehensive error handling, progress tracking, and result visualization
- **Cost-Optimized**: Multiple model options with different cost/accuracy trade-offs
- **Extensible**: Modular architecture for easy customization and extension

### Technical Highlights

- **58 Audio Features**: Tempo, spectral centroid, MFCCs, chroma, onset strength, RMS energy, and more
- **Natural Language Conversion**: Transforms numerical features into human-readable descriptions
- **Hybrid Prompting**: Combines multiple prompting techniques for optimal LLM performance
- **Real-time Progress Tracking**: Monitor classification progress with detailed logging
- **Comprehensive Analysis**: Generates confusion matrices, accuracy metrics, and detailed reports

## 🏗️ Architecture

### System Flow

```
┌─────────────┐
│ Audio File  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Feature Extraction  │ (58 features: tempo, spectral, MFCC, etc.)
└──────┬──────────────┘
       │
       ├──────────────────┐
       │                  │
       ▼                  ▼
┌──────────────┐  ┌──────────────────┐
│ CNN Approach │  │  LLM Approach    │
│              │  │                  │
│ Spectrogram  │  │ Natural Language │
│   Images     │  │   Description    │
│              │  │                  │
│   CNN Model  │  │  Advanced Prompt │
│              │  │                  │
│              │  │  AWS Bedrock     │
│              │  │  (Mixtral/Claude)│
└──────┬───────┘  └────────┬─────────┘
       │                   │
       └──────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Genre Prediction│
         │  + Confidence   │
         │  + Reasoning     │
         └─────────────────┘
```

### Component Architecture

#### 1. Audio Feature Extractor (`audio_feature_extractor.py`)
- Extracts 58 comprehensive audio features
- Handles multiple audio formats (WAV, AU)
- Robust error handling for corrupted files
- Statistical measures (mean, std, max) for each feature

#### 2. Feature Descriptor (`feature_descriptor.py`)
- Converts numerical features to natural language
- Multi-level descriptions (low, mid, high-level)
- Categorizes features (tempo, brightness, texture, energy)
- Provides genre suggestions based on features

#### 3. Prompt Generator (`prompt_generator.py`)
- Hybrid prompting strategy:
  - **Few-shot learning**: 5 examples of genre characteristics
  - **Chain-of-thought**: 6-step analysis process
  - **Multi-expert**: 3 expert perspectives (rhythm, frequency, energy)
  - **Confidence calibration**: Honest uncertainty assessment
- Generates 2000+ token prompts optimized for music classification

#### 4. Bedrock Client (`bedrock_client.py`)
- AWS Bedrock integration
- Supports multiple models (Mixtral, Claude Sonnet, Claude Haiku)
- Automatic retries with exponential backoff
- Rate limiting and error handling
- JSON response parsing and genre normalization

#### 5. Main Classifiers
- **`gtzan_cnn_classifier.py`**: CNN-based classification using spectrogram images
- **`bedrock_music_classifier.py`**: LLM-based classification using AWS Bedrock

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **AWS Account** (for LLM approach): With Bedrock access enabled
- **Kaggle Account** (for dataset): With API credentials configured
- **Git**: For cloning the repository

### Step 1: Clone the Repository

```bash
git clone https://github.com/mansigambhir-13/Music-Tagger.git
cd Music-Tagger
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

#### For Traditional ML/CNN Approach

```bash
cd gtzan_dataset
pip install -r requirements.txt
```

#### For LLM-Based Approach

```bash
cd gtzan_dataset
pip install -r requirements_bedrock.txt
```

#### Install All Dependencies

```bash
cd gtzan_dataset
pip install -r requirements.txt
pip install -r requirements_bedrock.txt
```

### Step 4: Configure Kaggle API

1. Sign up at [kaggle.com](https://www.kaggle.com)
2. Go to Account → Create API Token
3. Save `kaggle.json` to:
   - **Windows**: `C:\Users\YourUsername\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
4. Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

### Step 5: Configure AWS (For LLM Approach)

1. **Install AWS CLI**:
   ```bash
   pip install awscli
   ```

2. **Configure AWS Credentials**:
   ```bash
   aws configure
   # Enter: AWS Access Key ID, Secret Access Key, Region (us-east-1)
   ```

3. **Enable Bedrock Models**:
   - Go to [AWS Console → Bedrock](https://console.aws.amazon.com/bedrock/)
   - Navigate to "Model Access"
   - Enable:
     - Mixtral 8x7B Instruct
     - Claude 3.5 Sonnet
     - Claude 3 Haiku
   - Wait 1-2 minutes for access to activate

4. **Test Setup**:
   ```bash
   cd gtzan_dataset
   python test_bedrock_setup.py
   ```

## 🎬 Quick Start

### Option 1: Traditional ML/CNN Approach

```bash
cd gtzan_dataset

# Download dataset and run basic analysis
python gtzan_music_dataset.py

# Train CNN model
python gtzan_cnn_classifier.py
```

### Option 2: LLM-Based Approach (AWS Bedrock)

```bash
cd gtzan_dataset

# Test setup
python test_bedrock_setup.py

# Run classifier
python bedrock_music_classifier.py
```

### Quick Test with Single Sample

```bash
cd gtzan_dataset
python run_single_sample.py
```

## 📖 Usage

### Traditional ML/CNN Approach

#### Basic Analysis and Visualization

```bash
python gtzan_music_dataset.py
```

This script will:
- Download the GTZAN dataset (~1.2GB)
- Generate feature analysis plots
- Create audio visualizations
- Build and evaluate a Random Forest classifier
- Save visualizations to `./gtzan_visualizations/`

#### Deep Learning CNN Model

```bash
python gtzan_cnn_classifier.py
```

This script will:
- Load spectrogram images
- Build and train a CNN model
- Generate performance metrics
- Save results to `./gtzan_cnn_results/`

### LLM-Based Approach (AWS Bedrock)

#### Basic Usage

```bash
python bedrock_music_classifier.py
```

This script will:
1. Load GTZAN dataset
2. Extract features from audio files
3. Convert features to natural language descriptions
4. Generate advanced prompts
5. Invoke AWS Bedrock models
6. Tag each song with genre prediction
7. Generate comprehensive results and visualizations

#### With Progress Monitoring

```bash
# Run with detailed logging
python run_with_logs.py

# Monitor progress in real-time
python monitor_progress.py

# View live results
python show_live_results.py
```

### Custom Audio File Classification

You can classify your own audio files by modifying the scripts to accept custom file paths. See the documentation in each script for details.

## ⚙️ Configuration

### CNN Approach Configuration

Edit `gtzan_cnn_classifier.py`:

```python
# Image size
image_size = (256, 256)  # Adjust for different resolutions

# Model architecture
model = Sequential([
    Conv2D(64, (3, 3), ...),  # Modify filters, layers
    # Add more layers as needed
])

# Training parameters
epochs = 50
batch_size = 32
```

### LLM Approach Configuration

Edit `bedrock_music_classifier.py`:

```python
# Configuration (lines 523-525)
REGION = "us-east-1"  # AWS region
MODEL_NAME = "mixtral"  # Options: mixtral, claude_sonnet, claude_haiku
SAMPLES_PER_GENRE = 10  # Number of songs per genre
```

#### Available Models

| Model | Speed | Cost/Song | Accuracy | Best For |
|-------|-------|-----------|----------|----------|
| **Mixtral** | Fast | ~$0.0005 | 65-75% | Cost-effective, good balance |
| **Claude Sonnet** | Medium | ~$0.003 | 70-80% | Best accuracy |
| **Claude Haiku** | Very Fast | ~$0.00025 | 60-70% | Fastest, cheapest |

### Prompt Customization

Edit `prompt_generator.py` to customize:

```python
# Modify few-shot examples
FEW_SHOT_EXAMPLES = [
    # Add your own examples
]

# Adjust decision rules
DECISION_RULES = [
    # Add custom rules
]

# Change prompt structure
def create_hybrid_prompt(self, features, feature_desc):
    # Customize prompt generation
```

## 📊 Results & Performance

### Expected Accuracy

#### CNN Approach
- **Overall Accuracy**: 70-85%
- **Training Time**: 10-30 minutes (GPU dependent)
- **Model Size**: ~10MB
- **Best Genres**: Classical, Metal (>85%)
- **Challenging**: Rock vs Country overlap

#### LLM Approach

| Model | Overall Accuracy | Best Genres | Challenging Genres |
|-------|-----------------|-------------|-------------------|
| **Mixtral** | 65-75% | Metal, Classical, Reggae (>80%) | Rock vs Country (<60%) |
| **Claude Sonnet** | 70-80% | Metal, Classical, Reggae (>85%) | Rock vs Country (<65%) |
| **Claude Haiku** | 60-70% | Metal, Classical (>75%) | Rock vs Country (<55%) |

### Processing Speed

#### CNN Approach
- **Training**: 10-30 minutes (1000 samples)
- **Inference**: <1 second per song

#### LLM Approach
- **Per Song**: 3-6 seconds
- **100 Songs**: 5-10 minutes
- **1000 Songs**: 1-2 hours

### Cost Estimation (LLM Approach)

| Model | Cost per Song | 100 Songs | 1000 Songs |
|-------|--------------|-----------|------------|
| **Mixtral** | ~$0.0005 | ~$0.05 | ~$0.50 |
| **Claude Sonnet** | ~$0.003 | ~$0.30 | ~$3.00 |
| **Claude Haiku** | ~$0.00025 | ~$0.025 | ~$0.25 |

### Output Files

#### CNN Approach
```
gtzan_cnn_results/
├── model.h5                    # Trained model
├── training_history.json       # Training metrics
├── confusion_matrix.png        # Confusion matrix visualization
├── accuracy_plot.png          # Accuracy over epochs
└── classification_report.txt   # Detailed report
```

#### LLM Approach
```
bedrock_results/
├── tagged_songs.json           # All predictions with metadata
├── accuracy_metrics.json       # Accuracy statistics
├── confusion_matrix.csv        # Confusion matrix
├── classification_report.txt   # Detailed report
└── classification_results.png  # Visualizations
```

### Example Output

```json
{
  "file": "metal.00000.wav",
  "actual_genre": "metal",
  "predicted_genre": "metal",
  "confidence": 0.92,
  "is_correct": true,
  "reasoning": "The fast tempo (165 BPM), high energy (0.28 RMS), bright spectrum (3800 Hz), and highly percussive texture (0.18 ZCR) are all strong indicators of metal music. The aggressive rhythmic accents and intense energy level further confirm this classification.",
  "key_indicators": ["tempo", "energy", "spectrum", "texture"],
  "alternative_genres": ["rock", "punk"],
  "step1_tempo": "Fast tempo (165 BPM) suggests metal, rock, or disco",
  "step2_spectral": "Bright spectrum (3800 Hz) indicates metal or rock",
  "step3_texture": "Highly percussive (0.18 ZCR) suggests metal or hip-hop",
  "step4_energy": "Very high energy (0.28 RMS) indicates metal or rock",
  "step5_rhythm": "Strong rhythmic accents (0.75 onset) suggest metal",
  "step6_synthesis": "All features strongly point to metal genre",
  "expert1_rhythm": "Rhythm expert: Fast tempo and strong beats indicate metal",
  "expert2_frequency": "Frequency analyst: Bright spectrum confirms metal",
  "expert3_energy": "Energy expert: High energy level supports metal classification",
  "consensus": "All experts agree: metal is the most likely genre",
  "confidence_explanation": "High confidence due to strong alignment across all features"
}
```

## 📁 Project Structure

```
Music-Tagger/
├── README.md                          # This file
├── gtzan_dataset/                     # Main project directory
│   ├── README.md                      # Dataset-specific README
│   ├── README_BEDROCK.md             # Bedrock-specific documentation
│   ├── START_HERE.md                  # Quick start guide
│   ├── QUICK_START.md                 # Quick start for ML approach
│   ├── QUICK_START_BEDROCK.md         # Quick start for LLM approach
│   ├── SYSTEM_DESIGN.md               # System architecture
│   ├── PROJECT_SUMMARY.md             # Project overview
│   ├── EXECUTION_PLAN.md              # Execution plan
│   ├── KAGGLE_SETUP.md               # Kaggle setup guide
│   │
│   ├── Core Modules
│   ├── audio_feature_extractor.py    # Extract 58 audio features
│   ├── feature_descriptor.py          # Convert features to text
│   ├── prompt_generator.py            # Generate advanced prompts
│   ├── bedrock_client.py              # AWS Bedrock integration
│   │
│   ├── Classifiers
│   ├── gtzan_music_dataset.py         # Basic ML analysis
│   ├── gtzan_cnn_classifier.py        # CNN-based classification
│   ├── bedrock_music_classifier.py    # LLM-based classification
│   │
│   ├── Utilities
│   ├── download_dataset.py            # Download GTZAN dataset
│   ├── test_bedrock_setup.py          # Test AWS setup
│   ├── setup_aws_access.py            # AWS configuration helper
│   ├── run_single_sample.py           # Test single sample
│   ├── run_with_logs.py               # Run with detailed logging
│   ├── monitor_progress.py            # Monitor classification progress
│   ├── show_live_results.py           # View live results
│   ├── watch_progress.py              # Watch progress in real-time
│   ├── view_live_logs.py              # View live logs
│   ├── analyze_results.py             # Analyze classification results
│   │
│   ├── Results
│   ├── bedrock_results/               # LLM classification results
│   │   ├── tagged_songs.json
│   │   ├── accuracy_metrics.json
│   │   ├── confusion_matrix.csv
│   │   ├── classification_report.txt
│   │   └── classification_results.png
│   │
│   ├── Dependencies
│   ├── requirements.txt               # ML/CNN dependencies
│   ├── requirements_bedrock.txt       # LLM dependencies
│   └── setup.py                       # Setup script
│
└── music-tagger/                      # Research and notes
    ├── Paper Reviews/                 # Research papers
    ├── Notes.md                       # Project notes
    └── Ideas Tags.md                  # Ideas and tags
```

## 📚 Documentation

### Key Documents

1. **README.md** (This file) - Comprehensive project overview
2. **gtzan_dataset/START_HERE.md** - Quick start guide
3. **gtzan_dataset/README_BEDROCK.md** - Detailed Bedrock documentation
4. **gtzan_dataset/SYSTEM_DESIGN.md** - System architecture details
5. **gtzan_dataset/EXECUTION_PLAN.md** - Detailed execution plan
6. **gtzan_dataset/PROJECT_SUMMARY.md** - Project summary

### Code Documentation

All scripts include:
- Comprehensive docstrings
- Function descriptions
- Type hints
- Error handling documentation
- Usage examples

### API Reference

#### AudioFeatureExtractor

```python
extractor = AudioFeatureExtractor(sample_rate=22050, duration=30.0)
features = extractor.extract_features(audio_file)
# Returns: Dict with 58 audio features
```

#### FeatureDescriptor

```python
descriptor = FeatureDescriptor()
description = descriptor.create_feature_description(features)
# Returns: Natural language description of features
```

#### HybridPromptGenerator

```python
generator = HybridPromptGenerator()
prompt = generator.create_hybrid_prompt(features, feature_desc)
# Returns: Advanced prompt string (2000+ tokens)
```

#### BedrockClient

```python
client = BedrockClient(region="us-east-1")
response = client.classify_genre(prompt, model_name="mixtral")
# Returns: Genre prediction with confidence and reasoning
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Kaggle API Error

**Error**: `Could not find kaggle.json`

**Solution**:
- Ensure `kaggle.json` is in the correct location:
  - Windows: `C:\Users\YourUsername\.kaggle\kaggle.json`
  - Linux/Mac: `~/.kaggle/kaggle.json`
- Verify file permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

#### 2. AWS Credentials Not Configured

**Error**: `NoCredentialsError` or `Unable to locate credentials`

**Solution**:
```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region
```

#### 3. Model Access Denied

**Error**: `AccessDeniedException` or `Model access denied`

**Solution**:
1. Go to AWS Console → Bedrock → Model Access
2. Enable: Mixtral 8x7B, Claude 3.5 Sonnet, Claude 3 Haiku
3. Wait 1-2 minutes for activation
4. Verify with: `python test_bedrock_setup.py`

#### 4. Memory Error

**Error**: `MemoryError: Unable to allocate array`

**Solution**:
- Reduce batch size in CNN training
- Process fewer samples at once
- Use smaller image sizes
- Close other applications

#### 5. Import Error

**Error**: `ModuleNotFoundError: No module named 'librosa'`

**Solution**:
```bash
pip install -r requirements.txt
# or
pip install -r requirements_bedrock.txt
```

#### 6. Rate Limiting (AWS Bedrock)

**Error**: `ThrottlingException` or rate limit errors

**Solution**:
- Script includes automatic retries
- Reduce `SAMPLES_PER_GENRE` if persistent
- Add delays between requests
- Use Claude Haiku for faster processing

#### 7. Dataset Not Found

**Error**: `Dataset not found` or `FileNotFoundError`

**Solution**:
```bash
python download_dataset.py
# Or manually download from Kaggle
```

#### 8. Feature Extraction Errors

**Error**: `Feature extraction failed` or audio loading errors

**Solution**:
- Install librosa: `pip install librosa soundfile`
- Check audio file format (supports WAV, AU)
- Verify audio files are not corrupted
- Check file permissions

#### 9. GPU Not Available (CNN)

**Error**: `No GPU available, using CPU`

**Solution**:
- Install tensorflow-gpu: `pip install tensorflow-gpu`
- Use Google Colab for GPU access
- Training will be slower on CPU but still works

#### 10. JSON Parsing Errors (Bedrock)

**Error**: `JSON decode error` or invalid response format

**Solution**:
- The script includes JSON parsing with fallbacks
- Check model response format
- Try a different model (Mixtral vs Claude)
- Review prompt structure

### Getting Help

1. Check the troubleshooting section above
2. Review script comments and docstrings
3. Check the documentation files in `gtzan_dataset/`
4. Review AWS Bedrock documentation
5. Check Kaggle dataset discussion forum
6. Open an issue on GitHub

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow the existing code style
4. **Add tests**: If applicable, add tests for new features
5. **Update documentation**: Update README and docstrings
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**: Describe your changes clearly

### Code Style

- Follow PEP 8 Python style guide
- Use type hints for function parameters and returns
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Add error handling where appropriate

### Areas for Contribution

- **New Models**: Add support for additional LLM models
- **Feature Engineering**: Improve audio feature extraction
- **Prompt Optimization**: Enhance prompting strategies
- **Performance**: Optimize processing speed
- **Documentation**: Improve documentation and examples
- **Testing**: Add unit tests and integration tests
- **UI/UX**: Create web interface or GUI
- **Mobile App**: Develop mobile application

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The GTZAN dataset is available for academic research purposes. Please refer to the original dataset license for usage terms.

## 👥 Credits

### Project Author

- **Mansi Gambhir** - [@mansigambhir-13](https://github.com/mansigambhir-13)

### Dataset Credits

- **GTZAN Dataset**: George Tzanetakis
- **Kaggle Upload**: Andrada Olteanu
- **Dataset Paper**: [Marsyas Info](http://marsyas.info/downloads/datasets.html)

### Technology Stack

- **Audio Processing**: [Librosa](https://librosa.org/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/), [TensorFlow](https://www.tensorflow.org/)
- **AWS Services**: [AWS Bedrock](https://aws.amazon.com/bedrock/)
- **Data Analysis**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

### References

- [GTZAN Dataset Paper](http://marsyas.info/downloads/datasets.html)
- [Librosa Documentation](https://librosa.org/)
- [Music Information Retrieval](https://musicinformationretrieval.com/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Kaggle GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## 🌟 Acknowledgments

- Thanks to the open-source community for excellent tools and libraries
- Special thanks to George Tzanetakis for creating the GTZAN dataset
- AWS for providing Bedrock services
- All contributors and users of this project

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Open an issue](https://github.com/mansigambhir-13/Music-Tagger/issues)
- **Email**: [Your email if available]
- **Documentation**: Check the `gtzan_dataset/` directory for detailed docs

## 🚀 Future Enhancements

- [ ] Real-time audio streaming classification
- [ ] Web interface for easy classification
- [ ] Mobile app development
- [ ] Support for more audio formats
- [ ] Additional genre support
- [ ] Ensemble methods combining CNN and LLM approaches
- [ ] Transfer learning with pre-trained models
- [ ] Audio augmentation techniques
- [ ] Multi-label classification (multiple genres)
- [ ] Confidence-based filtering
- [ ] Batch processing API
- [ ] Docker containerization
- [ ] CI/CD pipeline

---

**Happy Music Genre Classification! 🎵🤖**

*Last Updated: 2024*
