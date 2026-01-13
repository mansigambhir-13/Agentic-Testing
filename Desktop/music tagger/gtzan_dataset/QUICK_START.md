# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Set Up Kaggle Credentials

**You need a Kaggle account and API token to download the dataset.**

1. **Create a Kaggle account** (if you don't have one):
   - Go to [https://www.kaggle.com](https://www.kaggle.com) and sign up (free)

2. **Get your API token**:
   - Log in to Kaggle
   - Click your profile → Account
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`

3. **Place the credentials file**:
   - **Windows**: `C:\Users\YourUsername\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - On Linux/Mac, set permissions: `chmod 600 ~/.kaggle/kaggle.json`

4. **Accept dataset terms**:
   - Visit: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
   - Click "Download" to accept terms (you don't need to actually download)

📖 **Detailed instructions**: See `KAGGLE_SETUP.md`

### Step 2: Install Dependencies

```bash
# Navigate to the project directory
cd gtzan_dataset

# Install all required packages
pip install -r requirements.txt
```

Or run the setup script:

```bash
python setup.py
```

### Step 3: Run the Scripts

#### Basic Analysis (Recommended first):

```bash
python gtzan_music_dataset.py
```

This will:
- ✅ Download the GTZAN dataset (~1.2GB)
- ✅ Analyze audio features
- ✅ Create visualizations
- ✅ Build a Random Forest classifier
- ✅ Save results to `./gtzan_visualizations/`

#### Deep Learning CNN (Optional):

```bash
python gtzan_cnn_classifier.py
```

This will:
- ✅ Load spectrogram images
- ✅ Train a CNN model
- ✅ Generate performance metrics
- ✅ Save results to `./gtzan_cnn_results/`

## 📋 What You Need

### Required:
- ✅ Python 3.8+
- ✅ Kaggle account (free)
- ✅ Kaggle API token (kaggle.json)
- ✅ Internet connection (for dataset download)

### Optional but Recommended:
- ✅ Virtual environment
- ✅ GPU (for faster CNN training)

## ⚠️ Important Notes

1. **First-time download**: The dataset is ~1.2GB, so the first run will take time to download
2. **Kaggle credentials**: Must be set up before running scripts
3. **Dataset terms**: Must accept terms on Kaggle website first
4. **Memory**: CNN training requires ~4-8GB RAM

## 🆘 Troubleshooting

### "Could not find kaggle.json"
→ Check `KAGGLE_SETUP.md` for credential setup

### "Dataset not found" or "Permission denied"
→ Accept dataset terms on Kaggle website first

### Import errors
→ Run: `pip install -r requirements.txt`

### Memory errors
→ Reduce batch size in CNN script or use smaller image size

## 📚 Next Steps

After running the scripts:

1. Explore the generated visualizations
2. Experiment with different features
3. Try modifying the CNN architecture
4. Check out the README.md for advanced options

---

**Ready to start?** Set up Kaggle credentials first, then run `python gtzan_music_dataset.py`!

