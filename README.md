README for FineTuning Mistral 7B Repository
Overview
This repository focuses on fine-tuning the Mistral 7B model using qLoRa, including scripts for preprocessing raw data and prompting a language model to generate specific prompts from given texts.

Key Files and Their Functions
finetune.py
Purpose: Fine-tuning Mistral7B
Features:
Training on 4x NVIDIA 4090 GPUs.
4-bit quantization using BitsAndBytes library.
Tokenization with left-side padding and EOS/BOS tokens.
Parallel processing support on multiple GPUs.
LoRa integration from the Peft library, targeting all linear layers.
Gradient checkpointing and WandB for logging.
prep_and_prompt.py
Purpose: Converts text files to JSONL.
Features:
Segments and tokenizes text files.
Generates prompts using LLaMA2-13B.
API polling for results.
Utilizes RunPod endpoint for LLaMA2-13B.
Employs two ThreadPoolExecutors with 30 workers each.
Processes executors simultaneously.
Outputs in JSONL format.
collect_data.py
Collects and preprocesses data, outputting a text file.
clean_jsonl.py
Cleans errors in the output file created by prep_and_prompt.py.
inference.py
Runs the fine-tuned model.
upload_model_hf.py
Uploads the model to HuggingFace.
