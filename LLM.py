import pandas as pd

# Read the Excel file
df = pd.read_excel('summaries.xlsx')

# Extract the summaries
summaries = df['Summary'].tolist()
import openai

openai.api_key = 'your-api-key'

def classify_summary(summary):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Categorize the following summary into one of the predefined categories: {summary}",
        max_tokens=10
    )
    return response.choices[0].text.strip()

# Classify each summary
categories = [classify_summary(summary) for summary in summaries]
# Add the categories to the DataFrame
df['Category'] = categories

# Write the DataFrame to a new Excel file
df.to_excel('categorized_summaries.xlsx', index=False)
