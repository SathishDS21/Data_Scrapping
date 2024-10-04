import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import re
from tqdm import tqdm  # For progress bar

# Download necessary resources for sentence tokenization
nltk.download('punkt')

def clean_text(text):
    """
    Clean the extracted text by removing unnecessary characters and extra spaces.
    """
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any unwanted characters, keeping only alphanumerics and basic punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

def get_article_summary(link):
    """
    Extract and summarize an article from the provided link.
    """
    try:
        # Send a GET request to fetch the article
        response = requests.get(link)
        response.raise_for_status()  # Check for any request errors

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract all text content from <p> tags (or other fallback tags if needed)
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])

        # Clean the extracted text
        text = clean_text(text)

        # If the text is empty, return a message
        if not text.strip():
            return "No content found."

        # Tokenize the text into sentences
        sentences = sent_tokenize(text)

        # Summarize the content by selecting the first few sentences (around 20-40 words)
        summary = []
        word_count = 0
        for sentence in sentences:
            words_in_sentence = len(sentence.split())
            if word_count + words_in_sentence <= 40:
                summary.append(sentence)
                word_count += words_in_sentence
            if word_count >= 20:  # Stop when we reach at least 20 words
                break

        # Join the selected sentences into a final summary
        return ' '.join(summary)

    except Exception as e:
        # Return error message if any exception occurs
        return f"Error fetching content: {str(e)}"

# Define file paths for input and output
input_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Scrapping Input data/Test_data_first.xlsx"
output_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Scrapping Output data/Data_syndicate.xlsx"

# Read the Excel file containing the links
df = pd.read_excel(input_file_path)

# Check if the 'Links' column exists in the Excel file
if 'Links' not in df.columns:
    raise ValueError("The Excel file does not contain a 'Links' column.")

# Initialize an empty list to store the summaries
summaries = []

# Process each link using tqdm for progress display
for link in tqdm(df['Links'], total=len(df), desc="Processing Articles"):
    summary = get_article_summary(link)
    summaries.append(summary)

# Add the summaries to the DataFrame
df['summary'] = summaries

# Save the DataFrame with summaries to the output Excel file
try:
    df.to_excel(output_file_path, index=False)
    print(f"Summaries saved to {output_file_path}\n")
except Exception as e:
    raise ValueError(f"Error saving output Excel file: {e}")

print("Processing complete!")