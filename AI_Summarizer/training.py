from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import json

# Loading the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', num_labels=2)

#Loading the dataset from a JSON file

f = open(f'/Users/uladzimircharniauski/Documents/GetBetterHead/AI_Summarizer/alpaca_gpt4_data.json')
data = json.load(f)
dataset_dict = {'question': [], 'context': [], 'answers': []}
for item in data:
    dataset_dict['question'].append(item['instruction'])
    dataset_dict['context'].append(item['input'])
    dataset_dict['answers'].append(item['output'])

dataset = Dataset.from_dict(dataset_dict)

#Loading the validation dataset from a JSON file - we use Vicuna evaluation datasets

f_question = open(f'/Users/uladzimircharniauski/Documents/GetBetterHead/AI_Summarizer/json_vicuna_questions.json')
f_answer = open(f'/Users/uladzimircharniauski/Documents/GetBetterHead/AI_Summarizer/json_vicuna_answers.json')

data_q = json.load(f_question)
data_answer = json.load(f_answer)

validation_dict = {'question': [], 'context': [], 'answers': []}

for item_q in data_q:
    validation_dict['question'].append(item_q['text'])
    validation_dict['context'].append(item_q['category'])

for item_a in data_answer:
    validation_dict['answers'].append(item_a['text'])

validation_dataset = Dataset.from_dict(validation_dict)

# Function to preprocess examples
def preprocess_function(examples):
    return tokenizer(examples["question"], examples["context"], examples["answers"], truncation=True, padding="max_length")

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_validation_datasets = validation_dataset.map(preprocess_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets, 
    eval_dataset = tokenized_validation_datasets
)

# Train the model
trainer.train()
trainer.evaluate()

print("BERT is trained and validated successfully")



