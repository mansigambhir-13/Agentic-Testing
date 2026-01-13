---
paper: "[[Advancing the Foundation Model for Music Understanding.pdf]]"
tags:
  - music-research
color-label:
  - Purple - References
  - Red - Main Idea
---

---
![[Advancing the Foundation Model for Music Understanding.pdf#page=3&rect=56,358,557,641|p.3]]
> [!PDF|important] [[Advancing the Foundation Model for Music Understanding.pdf#page=1&selection=150,0,155,56&color=important|p.1]]
> > MERT(Li et al. 2023)) integrated with large language models (LLMs) via lightweight adapters to overcome music-text data scarcity/
>

> [!PDF|important] [[Advancing the Foundation Model for Music Understanding.pdf#page=2&selection=1,0,5,15&color=important|p.2]]
> > MU-LLaMA(Liu et al. 2023b) (built on LLaMA(Touvron et al. 2023)) pioneered this approach using audio-adapted LLaMA layers and the MusicQA dataset (synthesized from captions and tags), demonstrating strong QA and captioning performance
> 
>

> [!PDF|important] [[Advancing the Foundation Model for Music Understanding.pdf#page=2&selection=5,17,10,41&color=important|p.2]]
> > MusiLingo(Deng et al. 2024) refined this paradigm by aligning frozen MERT embeddings with LLMs like Vicuna(Chiang et al. 2023) through a simple linear projector; its key contribution is the high-quality MusicInstruct (MI) dataset for instruction-tuning, enabling robust open-ended QA and outperforming MU-LLaMA.
> 
>

> [!PDF|important] [[Advancing the Foundation Model for Music Understanding.pdf#page=2&selection=11,0,14,10&color=important|p.2]]
> > LLARK(Gardner et al. 2023) employs an adapter-based architecture trained on augmented data to excel at instructionfollowing tasks, including detailed captioning and musical reasoning.
> 
>

> [!PDF|important] [[Advancing the Foundation Model for Music Understanding.pdf#page=2&selection=14,42,26,45&color=important|p.2]]
> >  M2UGen(Liu et al. 2023a) introduces a unified LLaMA 2-based framework combining comprehension (music QA, captioning) with cross-modal generation (text/image/video-to-music, editing), utilizing large synthetic instruction datasets (MUCaps, MUEdit etc.) and LoRA fine-tuning to achieve SOTA across both understanding and creative tasks.


> [!PDF|important] [[Advancing the Foundation Model for Music Understanding.pdf#page=2&selection=45,28,50,45&color=important|p.2]]
> > Qwen2-Audio(Chu et al. 2024) establishes a high standard as an audio-language model by integrating a Whisper-large-v3(Radford et al. 2023) encoder with a Qwen-7B(Bai et al. 2023) LLM through a refined three-stage training pipeline (pre-training, SFT, DPO) to achieve state-of-the-art performance.
 #music-paper 
>

> [!PDF|important] [[Advancing the Foundation Model for Music Understanding.pdf#page=2&selection=57,0,61,27&color=important|p.2]]
> > Kimi-Audio(KimiTeam et al. 2025) proposes a universal audio foundation model, built on a hybrid architecture and pretrained on over 13 million hours of diverse audio, aiming to unify perception, reasoning, and generation within a single, open-source framework.
#music-paper

> [!PDF|red] [[Advancing the Foundation Model for Music Understanding.pdf#page=2&selection=105,7,140,40&color=red|p.2]]
> > Our model is designed to accept an interleaved sequence of audio and text inputs and generate a coherent text output. Formally, for any input sequence of the form [A1, T1, A2, T2, . . . , An, Tn], where Ai represents an audio file and Ti a text segment, each modality is first transformed into a sequence of embedding vectors. These embedding sequences are then concatenated and fed into the language model to produce the final output. The overall architecture, depicted in Figure 1, comprises three core components: a language model backbone, an audio encoder, and a connector module to bridge the two modalities.
> 
>

> [!PDF|red] [[Advancing the Foundation Model for Music Understanding.pdf#page=2&selection=144,4,145,26&color=red|p.2]]
> > language model backbone is initialized from Qwen3- 8B-Base (Yang et al. 2025)
> 
>

> [!PDF|red] [[Advancing the Foundation Model for Music Understanding.pdf#page=3&selection=186,17,187,47&color=red|p.3]]
> > we initialize the encoder from Whisper-largev3 (Radford et al. 2023) as our audio backbone.
> 
>

> [!PDF|red] [[Advancing the Foundation Model for Music Understanding.pdf#page=3&selection=194,0,210,2&color=red|p.3]]
> > To create a comprehensive representation of the audio, we do not rely solely on the final output layer of the Whisper encoder. Instead, we adopt a multi-layer feature fusion strategy. Specifically, we extract the hidden states from four distinct layers of the encoder—layers 0, 7, 15, and 32—and concatenate them. This results in a rich feature vector with a dimension of 5120 (1280 × 4).
> 
>

> [!PDF|red] [[Advancing the Foundation Model for Music Understanding.pdf#page=3&selection=210,3,216,8&color=red|p.3]]
> > The rationale behind this approach is that different layers of a deep network capture different levels of abstraction. Early layers (e.g., layer 0) tend to preserve low-level acoustic details like timbre and pitch, while deeper layers (e.g., layer 32) capture more abstract, semantic information like melodic contours and rhythmic patterns
> 
>

> [!PDF|red] [[Advancing the Foundation Model for Music Understanding.pdf#page=3&selection=224,0,230,61&color=red|p.3]]
> > The Whisper encoder processes a 30-second audio clip into a sequence of 1500 embedding vectors, corresponding to a temporal frequency of 50 Hz. This high density of tokens can be computationally burdensome for the LLM and may not align well with the typical information density of text. To address this, we apply a temporal downsampling step. We use a 1D mean pooling layer with a kernel size and stride of
> 
>

> [!PDF|red] [[Advancing the Foundation Model for Music Understanding.pdf#page=4&selection=214,33,221,54&color=red|p.4]]
> > The long audio stream is first segmented into 30-second non-overlapping chunks. Each chunk is processed independently by the audio encoder and pooling layer. The resulting embedding sequences are then concatenated in their original order to form a single, continuous sequence representing the entire audio piece. This mechanism extends the model’s effective receptive field to any audio duration, enabling true song-level analysis.
> 
>

> [!PDF|red] [[Advancing the Foundation Model for Music Understanding.pdf#page=4&selection=227,7,233,12&color=red|p.4]]
> > Its purpose is to project the 5120-dimensional audio embeddings into the 4096-dimensional space of the Qwen3 language model. For this, we use a 2-layer Multilayer Perceptron (MLP). The MLP first expands the input dimension by a factor of two, applies a GELU non-linear activation function, and then projects it down to the target dimension of the LLM. 
> 
>

> [!PDF|note] [[Advancing the Foundation Model for Music Understanding.pdf#page=4&selection=263,26,265,7&color=note|p.4]]
> > Experiments are conducted using NVIDIA A100 40 GB GPUs, at most 16 ones across two nodes.
> 
>

> [!PDF|note] [[Advancing the Foundation Model for Music Understanding.pdf#page=4&selection=253,35,257,29&color=note|p.4]]
> > The entire training protocol, summarized in Table 2, is divided into two primary phases: a four-stage pre-training phase to build a robust foundation, and a dual-track fine-tuning phase to specialize the model for diverse MIR applications.

![[Screenshot 2025-10-27 at 3.19.13 PM.png]]

![[Pasted image 20251030184255.png]]
