#THIS PROGRAM IS TO CONVERT EXISTING QUESTION/ANSWER JSON LINE FILES INTO JSON. 

#PLEASE UTILIZE YOUR TRAINING DATASETS

import json

path_question = f'/Users/uladzimircharniauski/Documents/GetBetterHead/AI_Summarizer/question_vicuna-13b.jsonl'
path_answer = f'/Users/uladzimircharniauski/Documents/GetBetterHead/AI_Summarizer/answer_vicuna-13b.jsonl'

json_path_question = f'json_vicuna_questions.json'
json_path_answer = f'json_vicuna_answers.json'

def converter(jsonl_file_path, json_file_path):

    # List to store all parsed JSON objects
    json_list = []

    # Read the JSON Lines file
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each line as JSON and append to the list
            data = json.loads(line.strip())
            json_list.append(data)

    # Write the list of JSON objects to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)

    print(f"Converted {jsonl_file_path} to {json_file_path}")

if __name__ == "__main__":

    print("Convertion is in process. Please be patient.\n")

    converter(path_question,json_path_question)
    converter(path_answer,json_path_answer)