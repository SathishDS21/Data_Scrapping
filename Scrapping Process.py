import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import os

# Function to extract article content and generate summary
def get_article_summary(link):
    try:
        # Send a GET request to fetch the article
        response = requests.get(link)
        response.raise_for_status()  # Check for any request errors

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract all the text content from the webpage
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])

        # If the text is empty, return a message
        if len(text.strip()) == 0:
            return "No content found."

        # Tokenize the text into sentences
        sentences = sent_tokenize(text)

        # Summarize the content: We will take the first few sentences to form a summary of around 20-40 words.
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

# Define file paths
excel_file_path = "/Users/sathishm/Documents/TSM Folder/Scrapping Input data/Test_data_first.xlsx"
output_file_path = "/Users/sathishm/Documents/TSM Folder/Scrapping Output data/Test_data_output.xlsx"

# Read the Excel file containing the links
df = pd.read_excel(excel_file_path)

# Check if 'Links' column exists in the Excel file
if 'Links' not in df.columns:
    raise ValueError("The Excel file does not contain a 'Links' column.")

# Create a new column to store the summaries
df['summary'] = df['Links'].apply(get_article_summary)

# Save the updated DataFrame with summaries to a new Excel file
df.to_excel(output_file_path, index=False)

# Confirmation message
print(f"Summaries saved to {output_file_path}")