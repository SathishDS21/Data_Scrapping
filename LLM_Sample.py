import pandas as pd
from transformers import pipeline
from openpyxl import Workbook
import torch
device = 0 if torch.cuda.is_available() else -1
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()
categories = ["Business", "Technology", "Health", "Entertainment", "Politics", "Other"]
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
    # Read the input Excel file
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
    # Process each row in the Excel file
    for index, row in df.iterrows():
        try:
            text = row['Summary']  # Change 'Text' to the appropriate column name
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
if __name__ == "__main__":
    # Step 1: Define input and output Excel file paths
    input_file_path = "/Users/sathishm/Documents/TSM Folder/Scrapping Output data/Test_data_output.xlsx"
    output_file_path = "/Users/sathishm/Documents/TSM Folder/Scrapping Output data/Train_data.xlsx"
    # Step 2: Process the data and write the results to an output Excel file
    process_excel(input_file_path, output_file_path)