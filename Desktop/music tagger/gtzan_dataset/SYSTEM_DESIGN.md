# System Design: AWS Bedrock Music Genre Classification

## 🎯 Overview

A comprehensive system that uses AWS Bedrock LLMs to classify music genres by converting audio features into natural language descriptions. The system processes the GTZAN dataset and uses advanced prompting strategies for optimal accuracy.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GTZAN Dataset (1000 songs)                │
│             100 songs × 10 genres (blues, classical, ...)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Audio Feature Extraction Layer                  │
│  • Librosa: Tempo, Spectral Centroid, MFCCs, Chroma         │
│  • Onset Strength, Zero Crossing Rate, RMS Energy           │
│  • 58 features per song (30-second clips)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Feature → Natural Language Conversion               │
│  • Tempo: "165 BPM" → "very fast tempo (intense)"           │
│  • Spectral: "3800 Hz" → "bright and treble-focused"        │
│  • Energy: "0.28" → "very loud and intense"                 │
│  • Comprehensive multi-level description                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Advanced Prompting Strategy Engine                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Hybrid Strategy:                                    │   │
│  │  1. Few-Shot Examples (5 genres)                    │   │
│  │  2. Chain-of-Thought Reasoning                      │   │
│  │  3. Multi-Expert Analysis                           │   │
│  │  4. Confidence Calibration                          │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              AWS Bedrock Model Invocation                    │
│  Primary: mistral.mixtral-8x7b-instruct-v0:1                │
│  Backup:  anthropic.claude-3-5-sonnet-20241022-v2:0         │
│  Testing: anthropic.claude-3-haiku-20240307-v1:0            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              JSON Response Parsing                           │
│  {                                                           │
│    "genre": "metal",                                         │
│    "confidence": 0.92,                                       │
│    "reasoning": "Fast tempo + high energy + bright...",     │
│    "key_indicators": ["tempo", "energy", "spectrum"]        │
│  }                                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Results Aggregation & Analysis                    │
│  • Accuracy by model and genre                              │
│  • Confusion matrices                                       │
│  • Confidence vs accuracy correlation                       │
│  • Genre tagging for all songs                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Output Files                                    │
│  • tagged_songs.json (all predictions)                      │
│  • model_performance.csv (accuracy metrics)                 │
│  • confusion_matrices.png (visualizations)                  │
│  • classification_report.txt (detailed analysis)            │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### Stage 1: Audio Processing
1. Load audio file (30 seconds)
2. Extract 58 audio features using librosa
3. Calculate statistical measures (mean, std)
4. Generate feature dictionary

### Stage 2: Feature Description
1. Map numerical features to natural language
2. Create multi-level descriptions:
   - Low-level: Tempo, energy, brightness
   - Mid-level: Texture, rhythm, dynamics
   - High-level: Overall character, mood
3. Generate comprehensive text description

### Stage 3: Prompt Generation
1. Select best prompting strategy (hybrid approach)
2. Inject few-shot examples
3. Add chain-of-thought structure
4. Include multi-expert analysis
5. Add confidence calibration instructions

### Stage 4: LLM Invocation
1. Format prompt for specific model (Mixtral/Claude)
2. Invoke AWS Bedrock API
3. Parse JSON response
4. Extract genre, confidence, reasoning

### Stage 5: Results Processing
1. Compare prediction vs actual genre
2. Calculate accuracy metrics
3. Aggregate results by model/genre
4. Generate visualizations
5. Save tagged songs

## 📊 Key Components

### 1. Feature Extractor (`audio_feature_extractor.py`)
- Extracts 58 audio features
- Handles various audio formats (WAV, AU)
- Computes statistical measures
- Error handling for corrupted files

### 2. Feature Descriptor (`feature_descriptor.py`)
- Maps features to natural language
- Creates multi-level descriptions
- Handles edge cases (missing features)
- Generates consistent format

### 3. Prompt Generator (`prompt_generator.py`)
- Hybrid prompting strategy
- Few-shot examples selection
- Chain-of-thought structure
- Multi-expert analysis
- Confidence calibration

### 4. Bedrock Client (`bedrock_client.py`)
- AWS Bedrock integration
- Model selection and invocation
- Response parsing
- Error handling and retries
- Rate limiting

### 5. Results Analyzer (`results_analyzer.py`)
- Accuracy calculation
- Confusion matrix generation
- Performance visualization
- Statistical analysis
- Report generation

## 🎯 Prompting Strategy

### Hybrid Approach (Best of All)

```python
1. SYSTEM PROMPT
   - Define expert music analyst role
   - Set classification task
   - Specify output format (JSON)

2. FEW-SHOT EXAMPLES (5 examples)
   - Metal: Fast tempo, high energy, bright
   - Classical: Slow, smooth, dynamic range
   - Hip-hop: Moderate tempo, bass-heavy
   - Jazz: Swing rhythm, warm tone
   - Reggae: Slow, bass-forward, offbeat

3. CHAIN-OF-THOUGHT ANALYSIS
   - Step 1: Tempo analysis → genre candidates
   - Step 2: Spectral analysis → refine candidates
   - Step 3: Texture analysis → further refinement
   - Step 4: Energy analysis → final candidates
   - Step 5: Synthesis → final decision

4. MULTI-EXPERT PERSPECTIVE
   - Rhythm Expert: Tempo, beat patterns
   - Frequency Expert: Spectral content
   - Energy Expert: Dynamics, loudness
   - Consensus: Combine expert opinions

5. CONFIDENCE CALIBRATION
   - High confidence (0.9+): Clear indicators
   - Medium confidence (0.7-0.9): Some ambiguity
   - Low confidence (<0.7): Multiple possibilities
```

## 🔧 Configuration

### Model Selection
- **Primary**: Mixtral 8x7B (cost-effective, fast)
- **Backup**: Claude 3.5 Sonnet (highest accuracy)
- **Testing**: Claude 3 Haiku (fast, cheap)

### Processing Parameters
- **Samples per genre**: 10 (configurable)
- **Max retries**: 3
- **Rate limiting**: 10 requests/second
- **Timeout**: 30 seconds per request

### Feature Extraction
- **Duration**: 30 seconds per clip
- **Sample rate**: 22050 Hz
- **Features**: 58 total
- **MFCCs**: 13 coefficients

## 📈 Expected Performance

### Accuracy Targets
- **Mixtral**: 65-75% accuracy
- **Claude 3.5 Sonnet**: 70-80% accuracy
- **Claude 3 Haiku**: 60-70% accuracy

### Genre Performance
- **High (>80%)**: Metal, Classical, Reggae
- **Medium (60-80%)**: Hip-hop, Jazz, Disco
- **Low (<60%)**: Rock vs Country (overlap)

### Processing Speed
- **Feature extraction**: ~1 second per song
- **LLM inference**: ~2-5 seconds per song
- **Total**: ~3-6 seconds per song
- **1000 songs**: ~1-2 hours (with rate limiting)

## 💰 Cost Optimization

### API Call Reduction
- Batch processing
- Cache feature descriptions
- Limit samples per genre
- Use cheaper models first

### Model Selection
- Start with Mixtral (cheaper)
- Use Claude for validation
- Skip low-confidence predictions

### Caching Strategy
- Cache feature extractions
- Cache feature descriptions
- Cache LLM responses (if same features)

## 🚨 Error Handling

### Audio Processing Errors
- Skip corrupted files
- Log errors for review
- Continue processing

### API Errors
- Retry with exponential backoff
- Skip on persistent errors
- Log for analysis

### Parsing Errors
- Fallback to simple extraction
- Log malformed responses
- Use default values

## 📁 Output Structure

```
bedrock_results/
├── tagged_songs.json          # All predictions
├── model_performance.csv      # Accuracy metrics
├── confusion_matrices/
│   ├── mixtral_confusion.png
│   └── claude_confusion.png
├── visualizations/
│   ├── accuracy_comparison.png
│   ├── genre_performance.png
│   └── confidence_analysis.png
└── reports/
    ├── classification_report.txt
    └── detailed_analysis.json
```

## 🔒 Security & Best Practices

### AWS Credentials
- Use IAM roles (not hardcoded keys)
- Least privilege access
- Rotate credentials regularly

### Data Privacy
- No audio data stored in logs
- Only features and predictions
- Secure storage of results

### Rate Limiting
- Respect API limits
- Implement exponential backoff
- Monitor usage

## 🎯 Success Metrics

1. **Accuracy**: >70% overall accuracy
2. **Coverage**: 100% of songs processed
3. **Speed**: <5 seconds per song
4. **Cost**: <$10 for 1000 songs
5. **Reliability**: <1% error rate

## 🚀 Future Enhancements

1. **Ensemble Methods**: Combine multiple models
2. **Active Learning**: Focus on difficult cases
3. **Transfer Learning**: Fine-tune on GTZAN
4. **Real-time Classification**: Stream processing
5. **Multi-label Classification**: Multiple genres per song

