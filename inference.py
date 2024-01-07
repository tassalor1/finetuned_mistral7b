from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import torch



device = "auto"
model_path = "Tassalor1/SciFiMistral7B"          





bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    quantization_config=bnb_config if device == "auto" else None,
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")



prompt = '''System: You are the greatest sci-fi story author in the universe.
User: "Give me a short sci-fi story about rabbits."
Assistant: '''

limit = 1000

# tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")
if device != "cpu":
    inputs = inputs.to('cuda')

# gen the response from the model
output = model.generate(**inputs, 
                        temperature=0.8, 
                        do_sample=True, 
                        top_p=0.95, 
                        top_k=60, 
                        max_new_tokens=limit-len(inputs["input_ids"]),
                        pad_token_id=tokenizer.pad_token_id)

# decode the generated tokens 
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)