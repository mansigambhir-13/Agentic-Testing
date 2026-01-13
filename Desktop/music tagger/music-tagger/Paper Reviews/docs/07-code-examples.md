# Code Examples and Usage Patterns

This document provides practical code examples for using MuFun, from basic inference to training your own models. All examples are designed to be beginner-friendly with detailed explanations.

## Table of Contents

1. [Basic Inference](#basic-inference)
2. [Advanced Inference Patterns](#advanced-inference-patterns)
3. [Data Preparation](#data-preparation)
4. [Model Quantization](#model-quantization)
5. [Training and Fine-tuning](#training-and-fine-tuning)
6. [Reinforcement Learning](#reinforcement-learning)

---

## Basic Inference

### Installation Requirements

Before running inference, you need to install the required audio processing packages:

```bash
pip install transformers torch torchaudio mutagen
```

**Supported Audio Formats**: `.wav`, `.mp3`, `.flac`, `.opus`, `.ogg`

### Single Audio File Processing

The simplest way to use MuFun is to analyze a single audio file:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
hf_path = 'Yi3852/MuFun-Instruct'  # or 'Yi3852/MuFun-Base'
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False)
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(
    hf_path, 
    trust_remote_code=True, 
    torch_dtype="bfloat16"
)
model.to(device)

# Analyze a song
audio_path = "/path/to/your/song.mp3"
prompt = "\n<audio>Can you listen to this song and tell me its lyrics?"
response = model.chat(prompt=prompt, audio_files=audio_path, tokenizer=tokenizer)
print(response)
```

**How it works**:
- The `<audio>` tag in the prompt is a placeholder
- During inference, the audio file is processed and converted to embeddings
- These embeddings replace the `<audio>` tag in the sequence
- The model generates text based on both the audio and text context

**Example Output**:
```
The lyrics describe a melancholic story about lost love, 
with verses about walking alone in the rain and remembering 
better times...
```

### Multiple Audio Files

You can compare or analyze multiple songs in a single prompt:

```python
# Compare two songs
audio_files = ["/path/to/song1.mp3", "/path/to/song2.mp3"]
prompt = "\n<audio> This is song1. <audio> This is song2. Which song do you like more? Tell me the reason."
response = model.chat(prompt=prompt, audio_files=audio_files, tokenizer=tokenizer)
print(response)
```

**Important**: 
- Each `<audio>` tag corresponds to one audio file in order
- First `<audio>` → first file in the list
- Second `<audio>` → second file in the list

**Example Output**:
```
I prefer song2 because it has a more upbeat tempo and energetic 
rhythm. Song1 is slower and more melancholic, while song2 has 
a driving beat that makes it more engaging...
```

---

## Advanced Inference Patterns

### Segment-Based Analysis

Analyze only a specific time range of an audio file:

```python
# Analyze first 30 seconds
audio_path = "/path/to/your/song.mp3"
prompt = "\n<audio>How is the rhythm of this music clip?"
response = model.chat(
    prompt=prompt, 
    audio_files=audio_path, 
    segs=[0, 30.0],  # [start_time, end_time] in seconds
    tokenizer=tokenizer
)
print(response)
```

**Use Cases**:
- Analyze song intros: `segs=[0, 15]`
- Analyze choruses: `segs=[60, 90]`
- Compare different sections: Use multiple audio files with different segments

**Multiple Audio with Different Segments**:

```python
# Analyze intro of song1 and chorus of song2
audio_files = ["/path/to/song1.mp3", "/path/to/song2.mp3"]
segments = [[0, 30], [60, 90]]  # First 30s of song1, 60-90s of song2
prompt = "\n<audio> This is the intro. <audio> This is the chorus. Compare the energy levels."
response = model.chat(
    prompt=prompt, 
    audio_files=audio_files, 
    segs=segments,
    tokenizer=tokenizer
)
```

**Partial Segments**:
```python
# Only segment the second audio
segments = [None, [0, 30.0]]  # Full song1, first 30s of song2
```

### Available Models

MuFun has several variants for different tasks:

```python
# Base model - general music understanding
model_path = 'Yi3852/MuFun-Base'

# Instruct model - conversational music analysis
model_path = 'Yi3852/MuFun-Instruct'

# ACEStep model - specialized for music generation tasks
model_path = 'Yi3852/MuFun-ACEStep'

# ABC model - specialized for symbolic music (ABC notation)
model_path = 'Yi3852/MuFun-ABC'
```

### Text-Only Mode (Not Recommended)

While possible, using MuFun as a text-only model is not recommended:

```python
# This works but is not the intended use case
prompt = "What is the definition of blues music?"
response = model.chat(prompt=prompt, audio_files=None, tokenizer=tokenizer)
```

---

## Data Preparation

### Data Format

Training data is stored in JSON format with the following structure:

```json
{
    "id": "unique_identifier",
    "audio": "/path/to/audio/file.mp3",
    "conversations": [
        {
            "from": "human",
            "value": "<audio>\nWhat genre is this song?"
        },
        {
            "from": "gpt",
            "value": "This is a blues song."
        }
    ]
}
```

**Field Descriptions**:
- `id`: Unique identifier for the sample (use UUID or similar)
- `audio`: Absolute or relative path to the audio file
- `conversations`: List of conversation turns between human and assistant
  - `from`: Either "human" or "gpt"
  - `value`: The text content (use `<audio>` tag for audio placement)

### Creating Training Data from a Dataset

Here's a complete example using the GTZAN genre classification dataset:

```python
import datasets
from datasets import load_dataset
import shortuuid
import random
import json
from tqdm import tqdm
import os

# Load dataset
gtzan = load_dataset("marsyas/gtzan", "all").cast_column(
    "audio", 
    datasets.Audio(decode=False)
)
id2label_fn = gtzan["train"].features["genre"].int2str

# Prepare output
gtzan_data = []
output_path = 'adata/gtzan_train.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Question templates for variety
questions = [
    "What genre does this music track belong to?",
    "Can you tell me the genre of this musical piece?",
    "What type of music is this track classified as?",
    "What is the style of this particular music track?",
    "Could you identify the genre of this song?",
    "What category of music does this track fall under?",
    "What musical genre is associated with this track?",
    "How would you classify the genre of this music?",
    "What genre is this specific track of music?",
    "What kind of music genre does this track represent?"
]

# Additional context for the question
context = " (choose the genre from: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.)"

# Process each sample
for sample in tqdm(gtzan["train"]):
    sample_dict = {
        'id': shortuuid.uuid(),
        'audio': sample['audio']['path'],
        'conversations': [
            {
                "from": "human", 
                "value": "<audio>\n" + random.choice(questions) + context
            },
            {
                "from": "gpt", 
                "value": id2label_fn(sample["genre"])
            }
        ]
    }
    gtzan_data.append(sample_dict)

# Save to JSON
with open(output_path, 'w') as f:
    json.dump(gtzan_data, f, indent=4)

print(f"Created {len(gtzan_data)} training samples")
```

### Multi-Turn Conversations

For more complex interactions, you can have multiple conversation turns:

```python
sample = {
    "id": "multi_turn_example",
    "audio": "/path/to/song.mp3",
    "conversations": [
        {
            "from": "human",
            "value": "<audio>\nWhat genre is this?"
        },
        {
            "from": "gpt",
            "value": "This is a jazz song."
        },
        {
            "from": "human",
            "value": "What instruments can you hear?"
        },
        {
            "from": "gpt",
            "value": "I can hear a saxophone, piano, double bass, and drums."
        },
        {
            "from": "human",
            "value": "What is the mood of this piece?"
        },
        {
            "from": "gpt",
            "value": "The mood is relaxed and contemplative, with a slow tempo and smooth melodic lines."
        }
    ]
}
```

### Multiple Audio in Training Data

```python
sample = {
    "id": "comparison_example",
    "audio": ["/path/to/song1.mp3", "/path/to/song2.mp3"],
    "conversations": [
        {
            "from": "human",
            "value": "<audio> Song 1. <audio> Song 2. Which has a faster tempo?"
        },
        {
            "from": "gpt",
            "value": "Song 2 has a faster tempo with approximately 140 BPM compared to Song 1's 90 BPM."
        }
    ]
}
```

---

## Model Quantization

Quantization reduces memory usage by using lower precision numbers. This is essential for running large models on consumer GPUs.

### Memory Requirements

| Precision | VRAM Required | Song Length | Quality |
|-----------|---------------|-------------|---------|
| BF16 (native) | 24GB | Up to 5 minutes | Best |
| 8-bit | 16GB | Up to 5 minutes | Very Good |
| 4-bit | 12GB | Typical songs | Good |

**Note**: VRAM consumption scales with audio duration (approximately 10 tokens per second of audio).

### 4-bit Quantization

```python
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

hf_path = 'Yi3852/MuFun-Instruct'
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False)

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_skip_modules=["lm_head", "vision_tower", "connector"]
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    hf_path,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
    quantization_config=quantization_config
)

# Use normally
audio_path = "/path/to/song.mp3"
prompt = "\n<audio>What is the genre of this song?"
response = model.chat(prompt=prompt, audio_files=audio_path, tokenizer=tokenizer)
print(response)
```

**Quantization Parameters Explained**:
- `load_in_4bit=True`: Use 4-bit precision for weights
- `bnb_4bit_compute_dtype="bfloat16"`: Compute in BF16 for better accuracy
- `bnb_4bit_use_double_quant=True`: Further compress quantization constants
- `bnb_4bit_quant_type="nf4"`: Use NormalFloat4 quantization (best for neural networks)
- `llm_int8_skip_modules`: Don't quantize these critical modules

### 8-bit Quantization

For better quality with moderate memory savings:

```python
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=["lm_head", "vision_tower", "connector"]
)

model = AutoModelForCausalLM.from_pretrained(
    hf_path,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
    quantization_config=quantization_config
)
```

**When to use each**:
- **4-bit**: Limited VRAM (12GB), acceptable quality loss
- **8-bit**: More VRAM available (16GB), minimal quality loss
- **BF16**: Maximum VRAM (24GB+), best quality

---

## Training and Fine-tuning

### Installation

```bash
git clone https://github.com/laitselec/MuFun.git
cd MuFun

conda create -n mufun python=3.10 -y
conda activate mufun
pip install --upgrade pip
pip install -e .

# Optional: Install Flash Attention for faster training
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### Fine-tuning an Existing Model

Fine-tuning adapts a pre-trained model to your specific task or dataset.

**Step 1**: Prepare your data in JSON format (see [Data Preparation](#data-preparation))

**Step 2**: Configure the training script `scripts/finetune.sh`:

```bash
# Data paths
TRAIN_DATA_PATH="adata/train_data.json"
EVAL_DATA_PATH="adata/eval_data.json"
AUDIO_PATH="/"  # Root path for audio files

# Model configuration
LLM_VERSION="Qwen/Qwen3-8B-Base"
AT_VERSION="openai/whisper-large-v3"
CN_VERSION="blp_4i_2x"
CONV_VERSION="qwen2_instruct"
VERSION="exp1"
TRAIN_RECIPE="common"
MODEL_MAX_LENGTH=4096

deepspeed --include localhost:0,1,2,3,4,5 --master_port 29501 tinyllava/train/train.py \
    --deepspeed ./scripts/z3.json \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $AUDIO_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $AT_VERSION \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio audio \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower full \
    --tune_type_connector full \
    --pretrained_model_path Yi3852/MuFun-Base \
    --output_dir checkpoints/${VERSION}-ft1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 12 \
    --eval_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --tokenizer_use_fast False \
    --run_name mufun-${VERSION}-ft1
```

**Step 3**: Run the training:

```bash
sh scripts/finetune.sh
```

### Training Parameters Explained

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `deepspeed` | DeepSpeed config for distributed training | `./scripts/z2.json`, `./scripts/z3.json` |
| `data_path` | Path to training data JSON | `adata/train_data.json` |
| `eval_data_path` | Path to evaluation data JSON | `adata/eval_data.json` |
| `pretrained_model_path` | Initial model weights | `Yi3852/MuFun-Base` |
| `per_device_train_batch_size` | Batch size per GPU | 2-32 (adjust for VRAM) |
| `gradient_accumulation_steps` | Steps before updating weights | 2-12 (effective batch = batch_size × this) |
| `tune_type_llm` | Train or freeze LLM | `full`, `frozen`, `lora` |
| `tune_type_vision_tower` | Train or freeze Whisper | `full`, `frozen` |
| `tune_type_connector` | Train or freeze connector | `full`, `frozen` |
| `learning_rate` | Learning rate | `2e-5` to `5e-5` |
| `warmup_ratio` | Warmup proportion | `0.03` (3% of training) |
| `lr_scheduler_type` | Learning rate schedule | `cosine`, `linear` |
| `save_steps` | Save checkpoint every N steps | 30-100 |
| `save_total_limit` | Max checkpoints to keep | 1-3 |
| `output_dir` | Where to save checkpoints | `checkpoints/exp1` |
| `eval_steps` | Evaluate every N steps | 20-50 |
| `num_train_epochs` | Number of training epochs | 1-3 |
| `model_max_length` | Max sequence length | 4096-8192 |
| `gradient_checkpointing` | Save memory (slower) | `True` for large models |

**Effective Batch Size**:
```
Effective Batch Size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus
Example: 2 × 12 × 6 = 144
```

### Training from Scratch

Training from scratch involves two stages:

**Stage 1 - Warmup (Connector Only)**:

```bash
# Configure scripts/warmup_qwen.sh
# Key settings:
# - tune_type_llm: frozen
# - tune_type_vision_tower: frozen
# - tune_type_connector: full

sh scripts/train_scratch.sh  # This runs warmup_qwen.sh
```

**What happens**: Only the connector learns to translate Whisper features to LLM space. The Whisper encoder and Qwen3 LLM remain frozen.

**Stage 2 - Full Training**:

```bash
# Update scripts/trainfull_qwen.sh
# Set pretrained_model_path to the warmup checkpoint
# Key settings:
# - tune_type_llm: full
# - tune_type_vision_tower: full
# - tune_type_connector: full

# In train_scratch.sh, comment warmup line and uncomment full training line
sh scripts/train_scratch.sh  # This runs trainfull_qwen.sh
```

**What happens**: All components (Whisper + Connector + Qwen3) are trained together.

**Example Workflow**:

```bash
# Step 1: Run warmup training
sh scripts/train_scratch.sh

# Step 2: Wait for completion, note the checkpoint path
# Example: checkpoints/mufun-Qwen3-8B-whisper-large-v3-exp1-warmup/checkpoint-100

# Step 3: Edit trainfull_qwen.sh
# Change: --pretrained_model_path checkpoints/.../checkpoint-100

# Step 4: Edit train_scratch.sh
# Comment: # bash scripts/warmup_qwen.sh ...
# Uncomment: bash scripts/trainfull_qwen.sh ...

# Step 5: Run full training
sh scripts/train_scratch.sh
```

---

## Reinforcement Learning

MuFun supports GRPO (Group Relative Policy Optimization) for aligning the model with human preferences.

### Installation

```bash
# Install modified TRL library
cd trl-main/
conda install -c conda-forge pyarrow
pip install .
cd ..
```

### GRPO Training

**Step 1**: Prepare RL data in JSON format (similar to supervised data)

**Step 2**: Configure `scripts/grpo.sh`:

```bash
TRAIN_DATA_PATH="adata/rl_data.json"
EVAL_DATA_PATH="adata/eval_data.json"
AUDIO_PATH="/"
LLM_VERSION="Qwen/Qwen3-8B-Base"
AT_VERSION="openai/whisper-large-v3"
CN_VERSION="blp_4i_2x"
CONV_VERSION="qwen2_instruct"
VERSION="exp1"
MODEL_MAX_LENGTH=8192

deepspeed --include localhost:0,1,2,3,4,5 --master_port 29501 tinyllava/train/train_grpo.py \
    --deepspeed ./scripts/z2.json \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $AUDIO_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $AT_VERSION \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio audio \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_type_connector full \
    --pretrained_model_path Yi3852/MuFun-Instruct \
    --output_dir checkpoints/${VERSION}-RL1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 5 \
    --learning_rate 2e-6 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --model_max_length $MODEL_MAX_LENGTH \
    --save_steps 10 \
    --save_total_limit 1 \
    --report_to wandb \
    --run_name mufun-${VERSION}-RL1
```

**Step 3**: Run GRPO training:

```bash
sh scripts/grpo.sh
```

**Key Differences from Supervised Training**:
- Lower learning rate (`2e-6` vs `2e-5`)
- Vision tower typically frozen
- Requires a well-trained base model
- Uses reward-based optimization instead of cross-entropy loss

**When to use GRPO**:
- After supervised fine-tuning
- To align model outputs with human preferences
- To improve response quality and reduce hallucinations
- For task-specific optimization (e.g., better genre classification)

---

## Complete Training Pipeline Example

Here's a complete workflow from data preparation to deployment:

```python
# 1. Prepare training data
import json
import shortuuid

training_data = []
for audio_file, label in your_dataset:
    sample = {
        "id": shortuuid.uuid(),
        "audio": audio_file,
        "conversations": [
            {"from": "human", "value": "<audio>\nWhat genre is this?"},
            {"from": "gpt", "value": label}
        ]
    }
    training_data.append(sample)

with open('adata/train_data.json', 'w') as f:
    json.dump(training_data, f, indent=4)

# 2. Fine-tune the model
# sh scripts/finetune.sh

# 3. Load and test your fine-tuned model
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "checkpoints/exp1-ft1/checkpoint-100"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="bfloat16"
)
model.to("cuda")

# 4. Test inference
test_audio = "/path/to/test/song.mp3"
prompt = "\n<audio>What genre is this?"
response = model.chat(prompt=prompt, audio_files=test_audio, tokenizer=tokenizer)
print(f"Prediction: {response}")

# 5. Deploy with quantization for production
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_skip_modules=["lm_head", "vision_tower", "connector"]
)

production_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
    quantization_config=quantization_config
)
```

---

## Tips and Best Practices

### Inference Tips

1. **Audio Duration**: Keep songs under 5 minutes for BF16, or use quantization for longer audio
2. **Prompt Engineering**: Be specific in your questions for better responses
3. **Batch Processing**: Process multiple songs sequentially rather than in parallel to save VRAM
4. **Segment Analysis**: Use segments for long songs to focus on specific parts

### Training Tips

1. **Start Small**: Begin with a small dataset to verify your pipeline works
2. **Monitor Loss**: Use Weights & Biases (`--report_to wandb`) to track training
3. **Checkpoint Often**: Save checkpoints frequently in case of interruptions
4. **Gradient Accumulation**: Increase this if you run out of VRAM
5. **Learning Rate**: Start with `2e-5` and adjust based on validation loss
6. **Warmup**: Always use warmup (3% of training) for stable training
7. **Data Quality**: High-quality, diverse training data is more important than quantity

### Memory Optimization

1. **Gradient Checkpointing**: Enable for large models (saves VRAM, slower training)
2. **Mixed Precision**: Use BF16 (`--bf16 True`) for faster training
3. **DeepSpeed**: Use ZeRO-2 for training, ZeRO-3 for very large models
4. **Batch Size**: Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps`

### Common Issues

**Out of Memory**:
- Reduce batch size
- Enable gradient checkpointing
- Use DeepSpeed ZeRO-3
- Reduce `model_max_length`

**Slow Training**:
- Install Flash Attention
- Increase batch size (if VRAM allows)
- Use fewer dataloader workers
- Disable gradient checkpointing (if VRAM allows)

**Poor Results**:
- Check data quality and format
- Increase training epochs
- Adjust learning rate
- Ensure audio files are accessible
- Verify `<audio>` tags are correctly placed

---

## Summary

This document covered:
- **Basic and advanced inference** with single and multiple audio files
- **Data preparation** with practical examples
- **Model quantization** for deployment on consumer hardware
- **Training and fine-tuning** with detailed parameter explanations
- **Reinforcement learning** with GRPO for alignment
- **Best practices** for successful training and deployment

For more details on the architecture, see the other documentation files in the `docs/` folder.
