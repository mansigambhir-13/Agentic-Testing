# GTZAN Music Genre Classification Scripts

A comprehensive Python toolkit for working with the GTZAN music genre classification dataset from Kaggle.

## 📋 Overview

This repository contains Python scripts to:

- Download the GTZAN dataset from Kaggle
- Explore audio features and visualizations
- Build machine learning models for genre classification
- Train deep learning CNN models on spectrogram images

## 📁 Scripts

### 1. `gtzan_music_dataset.py`

Main script for dataset exploration and basic machine learning:

- Downloads the dataset automatically
- Explores CSV feature files
- Visualizes audio features
- Analyzes audio waveforms and spectrograms
- Builds a Random Forest classifier
- Generates comprehensive visualizations

### 2. `gtzan_cnn_classifier.py`

Advanced deep learning script for CNN-based classification:

- Loads and preprocesses spectrogram images
- Builds a Convolutional Neural Network
- Trains the model with callbacks
- Evaluates performance with detailed metrics
- Creates confusion matrices and accuracy plots

## 🚀 Quick Start

### Prerequisites

1. **Kaggle Account & API Key**

   - Sign up at [kaggle.com](https://www.kaggle.com)
   - Go to Account → Create API Token
   - Save `kaggle.json` to `~/.kaggle/` directory (or `C:\Users\YourUsername\.kaggle\` on Windows)
   - On Linux/Mac, set permissions: `chmod 600 ~/.kaggle/kaggle.json`

2. **Python Environment**

   - Python 3.8 or higher recommended
   - Virtual environment (optional but recommended)

### Installation

```bash
# Navigate to the project directory
cd gtzan_dataset

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Scripts

#### Basic Analysis and Visualization:

```bash
python gtzan_music_dataset.py
```

This will:

- Download the dataset (~1.2GB)
- Generate feature analysis plots
- Create audio visualizations
- Build and evaluate a Random Forest classifier
- Save all visualizations to `./gtzan_visualizations/`

#### Deep Learning CNN Model:

```bash
python gtzan_cnn_classifier.py
```

This will:

- Load spectrogram images
- Build and train a CNN model
- Generate performance metrics
- Save results to `./gtzan_cnn_results/`

## 📊 Dataset Structure

```
GTZAN Dataset/
├── genres_original/        # 1000 audio files (100 per genre)
│   ├── blues/
│   ├── classical/
│   ├── country/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/
├── images_original/        # Mel-spectrogram images
│   └── [same genre folders with .png files]
├── features_30_sec.csv     # Features for 30-second clips
└── features_3_sec.csv      # Features for 3-second clips
```

## 📈 Features Extracted

The dataset includes pre-extracted audio features:

- **Spectral Features**: centroid, bandwidth, rolloff, contrast
- **Temporal Features**: zero crossing rate, tempo
- **MFCCs**: Mel-frequency cepstral coefficients (20 coefficients)
- **Chroma Features**: 12 pitch classes
- **Statistical Measures**: mean and variance for each feature

## 🎯 Expected Results

### Random Forest Classifier

- Typical accuracy: 60-75%
- Best performing genres: Classical, Metal
- Challenging genres: Rock vs Country overlap

### CNN Model

- Typical accuracy: 70-85%
- Training time: 10-30 minutes (GPU dependent)
- Model size: ~10MB

## 🔧 Customization Options

### Modify Feature Selection

Edit `gtzan_music_dataset.py`:

```python
# Line ~250 - Select different features
key_features = ['mfcc1_mean', 'spectral_centroid_mean', ...]
```

### Adjust CNN Architecture

Edit `gtzan_cnn_classifier.py`:

```python
# Line ~180 - Modify layers
model = Sequential([
    Conv2D(64, (3, 3), ...),  # Change filters
    # Add more layers
])
```

### Change Image Size

```python
# Line ~70
image_size = (256, 256)  # Larger images
```

## 📝 Troubleshooting

### Common Issues

1. **Kaggle API Error**

   ```
   Error: Could not find kaggle.json
   ```

   Solution: Ensure kaggle.json is in `~/.kaggle/` (Linux/Mac) or `C:\Users\YourUsername\.kaggle\` (Windows)

2. **Memory Error**

   ```
   MemoryError: Unable to allocate array
   ```

   Solution: Reduce batch size or image size

3. **Import Error**

   ```
   ModuleNotFoundError: No module named 'librosa'
   ```

   Solution: Run `pip install -r requirements.txt`

4. **GPU Not Available**

   ```
   No GPU available, using CPU
   ```

   Solution: Install tensorflow-gpu or use Google Colab

## 🌟 Advanced Extensions

### Ideas for Improvement

1. **Data Augmentation**
   - Time stretching
   - Pitch shifting
   - Adding noise
   - Mix-up augmentation

2. **Advanced Models**
   - Transfer learning (VGG16, ResNet)
   - Attention mechanisms
   - Transformer models
   - Ensemble methods

3. **Feature Engineering**
   - Combine audio features with CNN features
   - Extract rhythm patterns
   - Use beat tracking
   - Harmonic-percussive separation

4. **Real-time Classification**
   - Build streaming audio classifier
   - Create web interface
   - Mobile app deployment

## 📚 References

- [GTZAN Dataset Paper](http://marsyas.info/downloads/datasets.html)
- [Librosa Documentation](https://librosa.org/)
- [Music Information Retrieval](https://musicinformationretrieval.com/)

## 📄 License

The GTZAN dataset is available for academic research purposes.
These scripts are provided as educational examples.

## 👥 Credits

- Dataset: George Tzanetakis
- Kaggle Upload: Andrada Olteanu
- Script Author: [Your Name]

## 💬 Support

For issues or questions:

1. Check the troubleshooting section
2. Review the script comments
3. Explore the Kaggle dataset discussion forum

---

Happy music genre classification! 🎵🤖

