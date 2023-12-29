import nltk
import torch
import requests
from text_generation import Client
import re
import tiktoken
nltk.download('punkt')

class prep_and_prompt():
    
    def __init__(self, data, url, api_key):
        self.data = data
        self.url = url
        self.api_key = api_key
        self.story = None
        self.segments = {}


        
    def load_data(self):
        with open(self.data, 'r', encoding='utf-8') as f:
            self.story = f.read()
    
    def clean_whitespace(self):
        self.story = re.sub(r'\n{2,}', '\n', self.story)
        self.story = re.sub(r' {2,}', '\n', self.story)
        self.story = self.story.strip()
        
    def encode_data(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.num_tokens = self.encoding.encode(self.story)
        
    def segment_text(self):
        self.segments = {}
        current_segment = []
        seg_num = 1
        
        # will loop untill segment has 1000 tokens
        for token in self.num_tokens:
            current_segment.append(token)
            
            # if >= 1000 will add this batch to dict
            if len(current_segment) >= 1000:
                segment_text = self.encoding.decode(current_segment)
                self.segments[f'Segment: {seg_num}'] = segment_text
                current_segment = []
                seg_num += 1
                
        # grabs remaining tokens        
        if current_segment:
            segment_text = self.encoding.decode(current_segment)
            self.segments[f'Segment: {seg_num}'] = segment_text
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
        "authorization": self.api_key
        }
        
        response = requests.post(self.url, json=payload, headers=headers)
    
        response_data = response.json()
    
        if response.status_code == 200:
            job_id = response_data.get('id')
            return job_id
        else:
            print("Failed to generate prompt:", response_data)
            return None
        
    def poll_for_result(self, job_id, interval=5):
        status_url = f"https://api.runpod.ai/v2/llama2-13b-chat/status/{job_id}"  
        headers = {
            "accept": "application/json",
            "authorization": self.api_key
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
            
    def extract_to_prompt(self):
        ''' 
        takes segmented input (dictionary) loops through,
        "gen_prompt" & "poll_for_result" functions
        takes llm gen prompt from output
        and places in new prompt with input text
        ''' 
        i = 1
        json_objects = []
        # loop through segmented outputs
        for _, text in self.segments.items():
            input_text = text
            job_id = self.generate_prompt(input_text)
            if job_id:
                output = self.poll_for_result(job_id)
                
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
                        json_objects.append(json_obj) 
            i += 1
        print(json_objects)
        return json_objects
    
    def execute(self):
        self.load_data()
        self.clean_whitespace()
        self.encode_data()
        self.segment_text()
        return self.extract_to_prompt()