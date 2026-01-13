# Quick Start Guide: AWS Bedrock Music Genre Classifier

## 🚀 5-Minute Setup

### Step 1: AWS Setup (2 minutes)

1. **Configure AWS CLI**
   ```bash
   aws configure
   # Enter: AWS Access Key ID, Secret Access Key, Region (us-east-1)
   ```

2. **Enable Bedrock Models**
   - Go to [AWS Console → Bedrock](https://console.aws.amazon.com/bedrock/)
   - Navigate to **Model Access**
   - Enable:
     - ✅ **Mistral Mixtral 8x7B Instruct** (`mistral.mixtral-8x7b-instruct-v0:1`)
     - ✅ **Claude 3.5 Sonnet** (`anthropic.claude-3-5-sonnet-20241022-v2:0`)
     - ✅ **Claude 3 Haiku** (`anthropic.claude-3-haiku-20240307-v1:0`)

### Step 2: Install Dependencies (1 minute)

```bash
cd gtzan_dataset
pip install -r requirements_bedrock.txt
```

### Step 3: Test Setup (1 minute)

```bash
python test_bedrock_setup.py
```

This will verify:
- ✅ AWS credentials
- ✅ Bedrock access
- ✅ Model access
- ✅ Feature extraction
- ✅ Prompt generation
- ✅ Simple inference

### Step 4: Run Classifier (1 minute to start)

```bash
python bedrock_music_classifier.py
```

The script will:
1. Load GTZAN dataset
2. Process 10 songs per genre (100 total)
3. Tag each song with genre prediction
4. Generate results and visualizations

## 📊 What You'll Get

### Output Files

```
bedrock_results/
├── tagged_songs.json          # All predictions with metadata
├── accuracy_metrics.json      # Accuracy statistics
├── confusion_matrix.csv       # Confusion matrix
├── classification_report.txt  # Detailed report
└── classification_results.png # Visualizations
```

### Example Results

```json
{
  "file": "metal.00000.wav",
  "actual_genre": "metal",
  "predicted_genre": "metal",
  "confidence": 0.92,
  "is_correct": true,
  "reasoning": "The fast tempo (165 BPM), high energy (0.28 RMS), bright spectrum (3800 Hz)...",
  "key_indicators": ["tempo", "energy", "spectrum"]
}
```

## 🎯 Configuration

### Quick Configuration

Edit `bedrock_music_classifier.py`:

```python
# Line 523-525
REGION = "us-east-1"  # AWS region
MODEL_NAME = "mixtral"  # Options: mixtral, claude_sonnet, claude_haiku
SAMPLES_PER_GENRE = 10  # Number of songs per genre
```

### Model Selection

- **Mixtral** (`mixtral`): Fast, cost-effective, 65-75% accuracy
- **Claude Sonnet** (`claude_sonnet`): Best accuracy, 70-80% accuracy
- **Claude Haiku** (`claude_haiku`): Fast and cheap, 60-70% accuracy

## 💰 Cost Estimation

### Per Song

- **Mixtral**: ~$0.0005 per song
- **Claude Sonnet**: ~$0.003 per song
- **Claude Haiku**: ~$0.00025 per song

### Example Costs

- 100 songs with Mixtral: ~$0.05
- 100 songs with Claude Sonnet: ~$0.30
- 100 songs with Claude Haiku: ~$0.025

## ⚡ Expected Performance

### Accuracy

- **Overall**: 65-75% (Mixtral), 70-80% (Claude Sonnet)
- **Best Genres**: Metal, Classical, Reggae (>80%)
- **Challenging**: Rock vs Country (<60%)

### Processing Speed

- **Per Song**: ~3-6 seconds
- **100 Songs**: ~5-10 minutes
- **1000 Songs**: ~1-2 hours

## 🚨 Troubleshooting

### Issue: "AWS credentials not configured"

**Solution:**
```bash
aws configure
# Or set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### Issue: "Model access denied"

**Solution:**
1. Go to AWS Console → Bedrock → Model Access
2. Enable required models
3. Wait 1-2 minutes for access to activate

### Issue: "Rate limit exceeded"

**Solution:**
- Script includes automatic retries
- If persistent, reduce `SAMPLES_PER_GENRE`
- Add delays between requests

### Issue: "Dataset not found"

**Solution:**
```bash
python download_dataset.py
```

## 📈 Next Steps

1. **Review Results**: Check `tagged_songs.json` for predictions
2. **Analyze Performance**: Review `classification_report.txt`
3. **Improve Accuracy**: Adjust prompts in `prompt_generator.py`
4. **Process More Songs**: Increase `SAMPLES_PER_GENRE`
5. **Try Different Models**: Test Claude Sonnet for better accuracy

## 🎵 Example Usage

### Process 5 songs per genre

```python
# In bedrock_music_classifier.py
SAMPLES_PER_GENRE = 5
```

### Use Claude Sonnet for better accuracy

```python
# In bedrock_music_classifier.py
MODEL_NAME = "claude_sonnet"
```

### Process all songs (100 per genre)

```python
# In bedrock_music_classifier.py
SAMPLES_PER_GENRE = 100
```

## 📚 Documentation

- **System Design**: See `SYSTEM_DESIGN.md`
- **Full Documentation**: See `README_BEDROCK.md`
- **Troubleshooting**: See `README_BEDROCK.md#troubleshooting`

## ✅ Verification Checklist

- [ ] AWS credentials configured
- [ ] Bedrock models enabled
- [ ] Dependencies installed
- [ ] Test script passes
- [ ] Dataset downloaded
- [ ] Ready to run classifier

---

**Ready to start?** Run `python test_bedrock_setup.py` to verify everything is configured correctly!

