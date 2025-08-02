# small_llm

## Multilingual 500M Parameter LLM  
*Custom Tokenizer & Supervised Fine-Tuning (SFT)*  
### Created during ISSAI summer program

---

## Overview

- transformer language model with language adapters, rotary position embeddings, and SwiGLU activation 
- custom sentencepiece unigram and byte-pairing encoding tokenizers 
- dataloaders for both supervised fine-tunning and base pretraining 
- training pipeline with pretraining on raw multilingual text and supervised fine-tuning on instruction-response pairs

---

## File Structure

```
small_llm/
│
├── dataloader/
│   ├── dataloader.py         # Streaming and SFT data loaders
│   └── sft_dataloader.py
│
├── inference/
│   └── inference.py          # Inference/generation script
│
├── llm_tokenizer/
│   ├── bpe_tokenizer.py      # BPE tokenizer training
│   ├── unigram_tokenizer.py  # Unigram tokenizer training
│   ├── create_tokenizer.py   # HuggingFace tokenizer wrapper
│   ├── test_tokenizers.py    # Tokenizer evaluation
│   ├── english_eval_texts.json
│   ├── kazakh_eval_texts.json
│   ├── russian_eval_texts.json
│   ├── spm_bpe_tokenizer_200m/
│   │   └── tokenizer_config.json
│   ├── smp_unigram_tokenizer/
│   │   └── tokenizer_config.json
│   └── spm_unigram_tokenizer_200m/
│       └── tokenizer_config.json
│
├── model/
│   └── model.py              # Main model architecture
│
├── train/
│   ├── train.py              # Pretraining script
│   └── sft_train.py          # SFT training script
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a Model
```bash
python train/train.py
```
Adjust config and paths as needed.

### 3. Fine-tune (SFT)
```bash
python train/sft_train.py
```

### 4. Run Inference
```bash
python inference/inference.py
```

---

## Model Access

**Pretrained 500M parameter model:** [sanzh-ts/llm_project](https://huggingface.co/sanzh-ts/llm_project/tree/main)

---

## Development

The entire project was developed in one month of daily coding, research, and reflection during the ISSAI summer program.

---

## License

MIT License
