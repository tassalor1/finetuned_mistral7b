FineTuning Mistral 7B Repository
Overview
This repository is dedicated to fine-tuning the Mistral 7B model using qLoRa. It also includes scripts for preprocessing raw data and generating specific prompts for language models from given texts.

Contents
1. finetune.py
Fine-tunes Mistral7B.
Features:
Training on 4x NVIDIA 4090 GPUs.
4-bit model quantization using BitsAndBytes.
Left-side padding and EOS/BOS token implementation.
Multi-GPU support.
LoRa integration from Peft library.
Gradient checkpointing.
WandB integration for logging.
2. prep_and_prompt.py
Converts text files to JSONL.
Features:
Text file segmentation and tokenization.
Prompt generation using LLaMA2-13B.
API polling for results.
RunPod endpoint for LLaMA2-13B.
Two ThreadPoolExecutors with 30 workers each.
Simultaneous processing of executors.
JSONL format output.
3. collect_data.py
Data collection and preprocessing.
4. clean_jsonl.py
Error cleaning in prep_and_prompt.py output.
5. inference.py
Runs the fine-tuned model.
6. upload_model_hf.py
Uploads the model to HuggingFace.
