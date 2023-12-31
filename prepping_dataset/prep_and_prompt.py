import nltk
import os
import requests
import json
import re
import tiktoken
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
nltk.download('punkt')



class prep_and_prompt():
    
    def __init__(self, folder_path, url, run_pod_api_key, hf_api_key, file_path, enable_logging):
        self.folder_path  = folder_path 
        self.url = url
        self.run_pod_api_key = run_pod_api_key
        self.hf_api_key = hf_api_key
        self.file_path = file_path
        self.enable_logging = enable_logging
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.story = None
        self.segments = {}
        self.files_processed = 0
        self.json_objects_created = 0
        self.json_objects_batch = []


    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.story = f.read()
        if self.enable_logging:
            print(f"Data loaded from file: {file_path}")

    
    def clean_whitespace(self):
        self.story = re.sub(r'\n{2,}', '\n', self.story)
        self.story = re.sub(r' {2,}', '\n', self.story)
        self.story = self.story.strip()

    def encode_data(self):
        self.num_tokens = self.tokenizer.encode(self.story)
        
  
    def segment_text(self):
        self.segments = {}
        current_segment = []
        seg_num = 1
        
        # sets random segment size for variations
        segment_size = random.randint(300, 1000)

        # will loop untill segment has specified segmetn_size 
        for token in self.num_tokens:
            current_segment.append(token)
            
            # if >= segment_size will add this batch to dict
            if len(current_segment) >= segment_size:
                segment_text = self.tokenizer.decode(current_segment)
                self.segments[f'Segment: {seg_num}'] = segment_text
                current_segment = []
                seg_num += 1
                segment_size = random.randint(300, 1000) # reset size 
                
        # grabs remaining tokens    
        if current_segment:
            segment_text = self.tokenizer.decode(current_segment)
            self.segments[f'Segment: {seg_num}'] = segment_text

        if self.enable_logging:
            print(f"{len(self.segments)} segments created")

        return self.segments

    def generate_prompt(self, input_text):
        
        prompt= f"""
        Based on the following story segment '{input_text}', directly create a brief sci-fi story prompt. 
        Start the prompt immediately without any introduction, explanation, or additional words. 
        End the prompt without any concluding remarks or questions. Provide only the prompt, exactly as requested, nothing more, nothing less.
        """

        payload = { "input": {
            "prompt": prompt,
            "sampling_params": {
                "max_tokens": 1000,
                "n": 1,
                "best_of": None,
                "presence_penalty": 0,
                "frequency_penalty": 0.2,
                "temperature": 0.6,
                "top_p": 1,
                "top_k": -1,
                "use_beam_search": False,
                "ignore_eos": False,
                "logprobs": None
            }
        } }
        headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": self.run_pod_api_key
        }
        
        response = requests.post(self.url, json=payload, headers=headers)
    
        response_data = response.json()
    
        if response.status_code == 200:
            job_id = response_data.get('id')
            return job_id
        else:
            print("Failed to generate prompt:", response_data)
            return None
        
    def poll_for_result(self, job_id, interval=1):
        ''' checks if request is waiting and will poll till it gets output '''
         
        status_url = f"https://api.runpod.ai/v2/llama2-13b-chat/status/{job_id}"  
        headers = {
            "accept": "application/json",
            "authorization": self.run_pod_api_key
        }
    
        while True:
            response = requests.get(status_url, headers=headers)
            result = response.json()
    
            if result['status'] == 'COMPLETED':
                output = result.get('output')
                return output  
            elif result['status'] in ['FAILED', 'ERROR']:
                print("Error or Failed Status:", result)
                return result
    
            time.sleep(interval)  # Wait before polling again

    
    def dump_jsonl(self, json_object):
        ''' 
        dumps segments into jsonl
        this is done is bacthes to speed things up
        '''
        with open(self.file_path, 'a', encoding='utf-8') as f:
            json_s = json.dumps(json_object)
            f.write(json_s + '\n')
        if self.enable_logging:
            print(f"Dumping {len(self.json_objects_batch)} JSON objects to file")

    def process_output(self, output, input_text):

        text = output['text'][0] if output['text'] else None
        extracted_text = ""
        
        # extract text between \n\n
        if text:
            matches = re.findall(r"\n\n(.*?)(?:\n\n|$)", text, re.DOTALL)
            if matches:
                extracted_text = matches[0].strip()
                json_obj = {
                "messages": [
                    {"role": "system", "content": "You are the greatest sci-fi story author in the universe."},
                    {"role": "user", "content": extracted_text},
                    {"role": "assistant", "content": input_text}
                ]
                }
                return json_obj
            
    def prompt_generation(self, job_queue) :     
        with ThreadPoolExecutor(max_workers=30) as executor:
            future_to_job_id  = {executor.submit(self.generate_prompt, text): text for _, text in self.segments.items()}
            for future in as_completed(future_to_job_id):
                segment = future_to_job_id[future]
                try:
                    job_id = future.result()
                    if job_id:
                        # store job_id with segment so we know what is what
                        job_queue.put((job_id, segment))
                except Exception as exc:
                    print(f'{segment} generated an exception: {exc}')

    def poll_prompt(self, job_queue, json_batch_size):
        total_processed = 0
        batch_start_time = None
        with ThreadPoolExecutor(max_workers=30) as executor:
            while True:
                try:
                    job_id, segment = job_queue.get(timeout=30)  # waits 30 secs for a job_id - if not will call queue.empty
                    if total_processed % json_batch_size == 0:  # Start timing at the beginning of a new batch
                        batch_start_time = time.time()
                    future = executor.submit(self.poll_for_result, job_id)
                    output = future.result()

                    if output:
                        json_proccess_obj = self.process_output(output, segment)
                        self.json_objects_batch.append(json_proccess_obj)

                        total_processed += 1  
                        if self.enable_logging and total_processed % 50 == 0:
                            print(f'Completed polling for {total_processed} segments')

                        # dumps segments once it hits specific size
                        if len(self.json_objects_batch) >= json_batch_size:
                            batch_end_time = time.time()
                            batch_execution_time = batch_end_time - batch_start_time
                            print(f"Time to process and dump {100} segments: {batch_execution_time:.2f} seconds")
                            self.dump_jsonl(self.json_objects_batch)
                            self.json_objects_batch = []

                except queue.Empty:
                    break  # exit if no jobs are left
                except Exception as exc: 
                    print(f'Polling for job {job_id} generated an exception: {exc}')

            # log the final count if it never reached 50 for that file
            if self.enable_logging and total_processed % 50 != 0:
                print(f'Completed polling for {total_processed} segments')

    def extract_to_prompt(self):
        json_batch_size = 100 
        job_queue = queue.Queue()

        # prompt generation in a separate thread
        prompt_thread = threading.Thread(target=self.prompt_generation, args=(job_queue,))
        prompt_thread.start()
        if self.enable_logging:
                    print(f"Prompt generation thread starting")

        # start polling in a separate thread
        time.sleep(20) # wait 20 secs so there is job_ids generated
        polling_thread = threading.Thread(target=self.poll_prompt, args=(job_queue, json_batch_size,))
        polling_thread.start()
        if self.enable_logging:
                    print(f"Polling prompt thread starting")

        # wait for both threads to complete
        prompt_thread.join()
        polling_thread.join()

        # dump any remaining json objects
        if self.json_objects_batch:
            self.dump_jsonl(self.json_objects_batch)
    

    def process_file(self, file_path):
        self.load_data(file_path)
        self.clean_whitespace()
        self.encode_data()
        self.segment_text()
        self.extract_to_prompt()
        if self.enable_logging:
            self.files_processed += 1

    def execute(self):
        ''' 
        checks if the entry is a file - if so calls 
        process_file method on each file
        '''
        for filename in os.listdir(self.folder_path):
            full_path = os.path.join(self.folder_path, filename)
            if os.path.isfile(full_path):
                if self.enable_logging:
                    print(f"Starting to process file: {filename}")
                self.process_file(full_path)
                if self.enable_logging:
                    print(f"Finished processing file: {filename}")
        if self.enable_logging:
            print(f"Total files processed: {self.files_processed}")
            print(f"Total JSON objects created: {self.json_objects_created}")