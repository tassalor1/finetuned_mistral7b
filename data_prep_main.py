# from prepping_dataset import collect_data
from prepping_dataset.prep_and_prompt import prep_and_prompt
from dotenv import load_dotenv
import os
load_dotenv() 

run_pod_api_key = os.environ.get("run_pod_api_key")
hf_api_key = os.environ.get("hf_api_key")


# collector = collect_data(pdf_file="D:\\coding\llms\\pdfs\\Beyond-the-Door.pdf",
#                          txt_file="D:\\coding\\llms\\sci_storys\\Beyond-the-Door.txt",
#                          page_start=4, 
#                          page_finish=12)
# collector.execute()


process = prep_and_prompt(folder_path="D:\coding\llms\sci_storys", 
                          url="https://api.runpod.ai/v2/llama2-13b-chat/runsync", 
                          run_pod_api_key=run_pod_api_key,
                          hf_api_key=hf_api_key,
                          file_path="D:\coding\llms\sci_fi_data2.jsonl",
                          enable_logging=True)  
process.execute()