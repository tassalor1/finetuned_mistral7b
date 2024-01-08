# FineTuning Mistral 7B Repository

## Overview
This repository is for fine-tuning the Mistral 7B model using qLoRa. It includes scripts for preprocessing raw data and generating specific prompts from llama-13b from given texts

## Files

### finetune.py
- Fine-tuning the Mistral7B model
- **Features**:
  - Training on 4x NVIDIA 4090 GPUs
  - Utilises BitsAndBytes library for 4-bit quantisation
  - Implements left side padding and EOS/BOS tokens in tokenisation
  - Supports parallel processing on multiple GPUs
  - Integrates LoRa from the Peft library, targeting all linear layers
  - Implements gradient checkpointing
  - Uses WandB for logging

### prep_and_prompt.py
- Converts text files to JSONL format
- **Features**:
  - Segments text files and tokenises the segments
  - Generates prompts using LLaMA2-13B
  - Implements API polling for result retrieval
  - Utilises RunPod endpoint for LLaMA2-13B
  - Uses two ThreadPoolExecutors with 30 workers each for parallel processing
  - Processes both executors simultaneously
  - Outputs generated JSON segments in JSONL format


#### collect_data.py
- Collects and preprocesses data, outputs as text file

#### clean_jsonl.py
- Cleans errors in output files from `prep_and_prompt.py`

#### inference.py
- Script to run the fine-tuned model

#### upload_model_hf.py
- Uploads the model to HuggingFace

