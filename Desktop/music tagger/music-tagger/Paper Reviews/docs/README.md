# MuFun Architecture Documentation

Welcome to the comprehensive documentation for MuFun, a multimodal foundation model that combines audio understanding with language generation. This documentation is designed to be beginner-friendly while providing deep technical insights for researchers and developers.

## What is MuFun?

MuFun is an AI system that can "listen" to music and "talk" about what it hears. It combines:
- **Whisper-large-v3** audio encoder (the "ears")
- **BLP Connector** (the "translator")
- **Qwen3-8B** language model (the "brain")

Together, these components enable MuFun to understand music at multiple levels and generate natural language descriptions, analyses, and responses.

## Documentation Structure

### Core Architecture Documents

1. **[00-overview.md](00-overview.md)** - Start Here!
   - High-level introduction to MuFun
   - The three-component architecture explained simply
   - How data flows through the system
   - Real-world examples and use cases
   - **Recommended for**: Everyone, especially beginners

2. **[01-audio-preprocessing.md](01-audio-preprocessing.md)** - Audio Preparation
   - How audio files are loaded and prepared
   - Resampling, chunking, and mel-spectrogram conversion
   - Dimension transformations with concrete examples
   - Supported audio formats and sampling rates
   - **Recommended for**: Understanding the input pipeline

3. **[02-whisper-encoder.md](02-whisper-encoder.md)** - Audio Understanding
   - How Whisper "listens" to audio
   - Multi-layer feature extraction strategy (layers 0, 7, 15, 32)
   - Pooling and concatenation operations
   - What each layer captures (acoustic → rhythmic → melodic → semantic)
   - **Recommended for**: Understanding audio feature extraction

4. **[03-connector.md](03-connector.md)** - Bridging Modalities
   - How the Connector "translates" between audio and text
   - BLP architecture layer-by-layer breakdown
   - Mathematical operations with actual numbers
   - Why the expansion-compression design works
   - **Recommended for**: Understanding multimodal fusion

5. **[04-qwen3-llm.md](04-qwen3-llm.md)** - Language Generation
   - How the language model processes audio + text
   - Text tokenization and embedding
   - Audio-text interleaving process
   - Causal attention and text generation
   - **Recommended for**: Understanding text generation

6. **[05-end-to-end-flow.md](05-end-to-end-flow.md)** - Complete Pipeline
   - Complete pipeline from audio file to text output
   - Comprehensive dimension tracking table
   - Worked example with a 3-minute blues song
   - Memory requirements and performance considerations
   - **Recommended for**: Seeing the complete picture

7. **[06-key-concepts.md](06-key-concepts.md)** - Concepts & Summary
   - Embeddings explained simply
   - Attention mechanism for beginners
   - Multimodal fusion concept
   - Causal generation process
   - Quick reference tables and architecture summary
   - **Recommended for**: Quick reference and concept review

8. **[07-code-examples.md](07-code-examples.md)** - Practical Usage
   - Inference code examples (single and multiple audio)
   - Data preparation formats and examples
   - Model quantization for deployment
   - Training and fine-tuning scripts
   - Reinforcement learning with GRPO
   - **Recommended for**: Developers and practitioners

## Recommended Reading Paths

### For Complete Beginners

If you're new to AI and want to understand MuFun from scratch:

1. Start with [00-overview.md](00-overview.md) to get the big picture
2. Read [01-audio-preprocessing.md](01-audio-preprocessing.md) to understand audio preparation
3. Read [02-whisper-encoder.md](02-whisper-encoder.md) to see how audio features are extracted
4. Read [03-connector.md](03-connector.md) to understand the translation step
5. Read [04-qwen3-llm.md](04-qwen3-llm.md) to see how text generation works
6. Read [05-end-to-end-flow.md](05-end-to-end-flow.md) to see everything connected
7. Read [06-key-concepts.md](06-key-concepts.md) for a summary and quick reference
8. Explore [07-code-examples.md](07-code-examples.md) when you're ready to use MuFun

**Estimated reading time**: 2-3 hours for complete understanding

### For Developers

If you want to use MuFun in your projects:

1. Skim [00-overview.md](00-overview.md) for context
2. Jump to [07-code-examples.md](07-code-examples.md) for practical usage
3. Read [05-end-to-end-flow.md](05-end-to-end-flow.md) for the complete pipeline
4. Refer to [06-key-concepts.md](06-key-concepts.md) for quick reference
5. Dive into specific components (01-04) as needed for debugging or optimization

**Estimated reading time**: 1 hour to get started, reference as needed

### For Researchers

If you're researching multimodal AI or music understanding:

1. Read [00-overview.md](00-overview.md) for the architecture overview
2. Read [02-whisper-encoder.md](02-whisper-encoder.md) for multi-layer feature extraction
3. Read [03-connector.md](03-connector.md) for the projection mechanism
4. Read [04-qwen3-llm.md](04-qwen3-llm.md) for multimodal fusion details
5. Read [05-end-to-end-flow.md](05-end-to-end-flow.md) for dimension tracking
6. Refer to [06-key-concepts.md](06-key-concepts.md) for technical specifications
7. Review [07-code-examples.md](07-code-examples.md) for training procedures

**Estimated reading time**: 2-3 hours for deep understanding

### Quick Reference

If you just need specific information:

- **Model specifications**: [06-key-concepts.md](06-key-concepts.md#quick-reference-tables)
- **Memory requirements**: [05-end-to-end-flow.md](05-end-to-end-flow.md#memory-requirements)
- **Code examples**: [07-code-examples.md](07-code-examples.md)
- **Dimension tracking**: [05-end-to-end-flow.md](05-end-to-end-flow.md#complete-dimension-tracking-table)
- **Training parameters**: [07-code-examples.md](07-code-examples.md#training-parameters-explained)

## Key Features of This Documentation

### Beginner-Friendly

- **Simple analogies**: Complex concepts explained using everyday examples
- **Visual diagrams**: Mermaid flowcharts showing data flow and transformations
- **Concrete examples**: Real numbers and calculations at every step
- **Progressive complexity**: Start simple, gradually add detail
- **No jargon**: Technical terms are always explained

### Comprehensive

- **Complete coverage**: Every component explained in detail
- **Dimension tracking**: Exact shapes and sizes at every stage
- **Worked examples**: Full 3-minute song traced through the pipeline
- **Code examples**: Practical usage patterns and training scripts
- **Performance data**: Memory requirements, timing, and optimization tips

### Accurate

- **Verified numbers**: All dimensions and calculations are correct
- **Consistent terminology**: Same terms used throughout all documents
- **Cross-referenced**: Links between related sections
- **Up-to-date**: Reflects the current MuFun implementation

## Quick Start

**Want to use MuFun right now?**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model_path = 'Yi3852/MuFun-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype="bfloat16"
).to("cuda")

# Analyze a song
audio_path = "/path/to/your/song.mp3"
prompt = "\n<audio>What genre is this song?"
response = model.chat(prompt=prompt, audio_files=audio_path, tokenizer=tokenizer)
print(response)
```

For more examples, see [07-code-examples.md](07-code-examples.md).

## Technical Specifications

| Property | Value |
|----------|-------|
| **Total Parameters** | 9.6 billion |
| **Audio Encoder** | Whisper-large-v3 (1.5B params) |
| **Connector** | BLP_4i_2x (100M params) |
| **Language Model** | Qwen3-8B (8B params) |
| **Embedding Dimension** | 3584 |
| **Context Length** | 4096 tokens |
| **Audio Token Rate** | ~10 tokens per second |
| **Supported Audio** | MP3, WAV, FLAC, OGG, M4A |
| **VRAM (BF16)** | ~21 GB |
| **VRAM (4-bit)** | ~7 GB |

## Architecture Overview

```
🎵 Audio File → 👂 Whisper Encoder → 🔄 Connector → 🧠 Qwen3 LLM → 💬 Text Response
```

**Three main components**:
1. **Whisper Encoder**: Extracts multi-level audio features (acoustic, rhythmic, melodic, semantic)
2. **Connector**: Projects audio features to language model embedding space
3. **Qwen3 LLM**: Processes combined audio-text sequence and generates responses

## Common Use Cases

- **Genre Classification**: "What genre is this song?"
- **Music Description**: "Describe the mood and instruments in this piece"
- **Lyric Recognition**: "What are the lyrics of this song?"
- **Music Comparison**: "Compare these two songs and tell me which is more upbeat"
- **Segment Analysis**: "How is the rhythm in the first 30 seconds?"
- **Instrumentation**: "What instruments can you hear?"
- **Tempo Analysis**: "What is the tempo of this song?"
- **Mood Detection**: "What emotions does this music convey?"

## System Requirements

### Minimum (4-bit quantization)
- GPU: NVIDIA RTX 3080 (10GB) or better
- RAM: 16 GB
- Storage: 10 GB for model weights
- Audio: Up to 3-minute songs

### Recommended (BF16 precision)
- GPU: NVIDIA A100 (40GB) or A6000 (48GB)
- RAM: 32 GB
- Storage: 25 GB for model weights
- Audio: Up to 5-minute songs

### For Training
- GPU: 4-8× NVIDIA A100 (40GB or 80GB)
- RAM: 128 GB+
- Storage: 100 GB+ for checkpoints and data

## Contributing

Found an error or have suggestions for improving the documentation? Please open an issue or pull request on the [MuFun GitHub repository](https://github.com/laitselec/MuFun).

## Citation

If you use MuFun in your research, please cite:

```bibtex
@article{mufun2024,
  title={MuFun: A Multimodal Foundation Model for Music Understanding},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This documentation is part of the MuFun project. See the main repository for license information.

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/laitselec/MuFun/issues)
- **Documentation**: You're reading it!
- **Examples**: See [07-code-examples.md](07-code-examples.md)

---

**Ready to dive in?** Start with [00-overview.md](00-overview.md) to begin your journey through MuFun's architecture!


![[Pasted image 20251029230750.png]]
