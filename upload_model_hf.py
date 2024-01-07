import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
load_dotenv() 


hf_api_key = os.environ.get("hf_api_key")
repo_name =
local_model_path = 
device = 'auto'



model = AutoModelForCausalLM.from_pretrained(
    local_model_path, 
    trust_remote_code=True, 
    device_map=device, 
    torch_dtype=torch.float16,
).eval()



tokenizer = AutoTokenizer.from_pretrained(local_model_path)



model.push_to_hub(repo_name, token=hf_api_key)
tokenizer.push_to_hub(repo_name, token=hf_api_key)