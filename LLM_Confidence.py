import pandas as pd
from transformers import pipeline
from openpyxl import Workbook
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
device = 0 if torch.cuda.is_available() else -1
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()
categories = ["Business", "Technology", "Health", "Entertainment", "Politics", "Other", "Culture", "Geography", "History", "Religion", "Education"]

def train_summarizer(train_data):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    inputs = tokenizer([item['text'] for item in train_data], max_length=1024, truncation=True, padding=True,
                       return_tensors="pt")
    labels = tokenizer([item['summary'] for item in train_data], max_length=128, truncation=True, padding=True,
                       return_tensors="pt")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs,
        eval_dataset=labels
    )
    trainer.train()
def summarize_text(text):
    try:
        summarized = summarizer(text, max_length=10, min_length=5, do_sample=False, clean_up_tokenization_spaces=True)
        return summarized[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return ""
def categorize_text(text):
    try:
        result = classifier(text, candidate_labels=categories)
        return result['labels'][0]
    except Exception as e:
        print(f"Error categorizing text: {e}")
        return "Uncategorized"
def process_excel(input_file_path, output_file_path):
    try:
        df = pd.read_excel(input_file_path)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    output_data = {
        'Original Text': [],
        'Summary': [],
        'Category': []
    }
    for index, row in df.iterrows():
        try:
            text = row['Summary']
            summary = summarize_text(text)
            category = categorize_text(summary)
            output_data['Original Text'].append(text)
            output_data['Summary'].append(summary)
            output_data['Category'].append(category)
        except KeyError:
            print(f"Column 'Text' not found in the input file.")
            return
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    try:
        output_df = pd.DataFrame(output_data)
        output_df.to_excel(output_file_path, index=False)
        print(f"Output written to {output_file_path}")
    except Exception as e:
        print(f"Error writing output file: {e}")
def categorize_text(text, threshold=0.1):
    try:
        result = classifier(text, candidate_labels=categories)
        if result['scores'][0] >= threshold:
            return result['labels'][0]  # Return the category with the highest score
        else:
            return "Uncategorized"  # Return "Uncategorized" if confidence is low
    except Exception as e:
        print(f"Error categorizing text: {e}")
        return "Uncategorized"
if __name__ == "__main__":
    input_file_path = "/Users/sathishm/Documents/TSM Folder/Scrapping Output data/Test_data_output.xlsx"
    output_file_path = "/Users/sathishm/Documents/TSM Folder/Scrapping Output data/Train_data.xlsx"
    process_excel(input_file_path, output_file_path)