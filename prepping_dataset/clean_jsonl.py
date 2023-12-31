import json
''' 
the way i appended jsons to the jsonl file was not correct as it has nulls in between jsons and is within a list.
this cleans this up for a correct jsonl file 
'''
def process_jsonl_line(line):
    try:
       
        data = json.loads(line)

        # check if a list - filter out nulls and return if not empty
        if isinstance(data, list):
            data = [item for item in data if item is not None]
            if data:
                return data

        # check if single json file
        elif data is not None:
            return data

    except json.JSONDecodeError:
        # Skip lines that are not valid JSON
        pass

    return None

def clean_jsonl_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in lines:
            processed_data = process_jsonl_line(line)

          
            if processed_data:
                # if the processed data is a list write each item on a new line
                if isinstance(processed_data, list):
                    for item in processed_data:
                        json.dump(item, outfile)
                        outfile.write('\n')
                else:
                    json.dump(processed_data, outfile)
                    outfile.write('\n')

input_file_path = "D:\\coding\\llms\\sci_fi_data2.jsonl"
output_file_path = "D:\\coding\\llms\\format_sci_fi_data2.jsonl"

clean_jsonl_file(input_file_path, output_file_path)