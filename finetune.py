from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb, os
from datetime import datetime
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,TrainingArguments, Trainer
import torch


wandb_project = "mistral7b-finetune"
wandb.login()

base_model_id = "mistralai/Mistral-7B-v0.1"

data_set = load_dataset('json', data_files="format_sci_fi_data2.jsonl", split='train')
max_length= 1000
lora_target_modules = [    # which layers to apply lora to
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]


lora_dropout = 0.05 # Dropout for lora weights to avoid overfitting
lora_bias = "none"
lora_r=32 # Bottleneck size between A and B matrix for lora params
lora_alpha=64 # how much to weigh LoRA params over pretrained params


project = "sci-fi-finetune"
base_model_name = "mistral7b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name


warmup_steps=1
per_device_train_batch_size=4
gradient_accumulation_steps=1
max_steps=500
learning_rate=0.00005
bf16=True
optim_type = "paged_adamw_8bit" # optimizer
logging_steps=25             # When to start reporting loss
logging_dir="./logs"        
save_strategy="steps"       # Save the model checkpoint every logging step
save_steps=25                # Save checkpoints every 25 steps
evaluation_strategy="steps" # Evaluate the model every logging step
eval_steps=25               # Evaluate and save checkpoints every 25 steps
report_to="wandb"          
run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"    



''' 
use 4bit quantization on model 
use of double quant to try and retain the loss info
from downsizing 32bit to 4bit
'''
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map='auto')



''' define tokenizer with padding and eos/ bos tokens'''
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token



''' tokenize dataset and pad all data to length 1000 tokens '''
def generate_and_tokenize_prompt(entry):
    combined_messages = ' '.join(f"[{message['role']}] {message['content']}" for message in entry['messages'])
    tokenized_entry = tokenizer(
        combined_messages,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokenized_entry["labels"] = tokenized_entry["input_ids"].copy()
    return tokenized_entry

tokenized_data_set = data_set.map(generate_and_tokenize_prompt)


t_data_set = tokenized_data_set.shuffle()

train_size = int(len(t_data_set) * 0.8)
test_size = len(t_data_set) - train_size
data_train = t_data_set.select(range(train_size))
data_test = t_data_set.select(range(train_size, train_size + test_size))


# qLoRa
model.gradient_checkpointing_enable() # reduces memory usage
model = prepare_model_for_kbit_training(model)



config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    bias=lora_bias,
    lora_dropout=lora_dropout, 
    task_type="CAUSAL_LM",
)


model = get_peft_model(model, config)
print_trainable_parameters(model)



# if more than 1 gpu
if torch.cuda.device_count():
    model.is_parallelizable = True
    model.model_parallel = True



training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_steps=max_steps,
        learning_rate=learning_rate, 
        bf16=True,
        optim=optim_type,
        logging_steps=logging_steps,             
        logging_dir=logging_dir,     
        save_strategy=save_strategy,      
        save_steps=save_steps,               
        evaluation_strategy=evaluation_strategy, 
        eval_steps=eval_steps,            
        do_eval=True,         # perform evaluation at the end of training
        report_to="wandb",           
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"    
    
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_test,
    tokenizer=tokenizer,
)

trainer.train()