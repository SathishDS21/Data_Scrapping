from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
from tqdm import tqdm

# Load the Flan-T5-large model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Function to summarize an article based on category
def summarize_article(article_text, category_prompt):
    # Create prompt with the category for context
    prompt = f"Summarize this article related to {category_prompt}: {article_text}"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    # Generate summary
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=15, length_penalty=6.0, num_beams=5,
                                 early_stopping=True)
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


# Function to predict supply chain impact based on content and category
def call_flan_t5_large(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt = prompt,
    return response.strip()



def predict_supply_chain_impact(content, category):
    prompt = (
        f"Based on the category '{category}' and the following content, identify the supply chain area affected and explain briefly why.\n\n"
        f"Content: {content}\n\n"
        f"Respond with a single sentence describing the affected supply chain area and a reason for impact."
    )
    response = call_flan_t5_large(prompt)

    if ":" in response:
        impact_area, reason = response.split(":", 1)
        return impact_area.strip(), reason.strip()[:150]
    else:
        return response.strip(), ""


# Load the Excel data
file_path = '/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM data/LLM Output.xlsx'  # Update with your actual file path
data = pd.read_excel(file_path)

# Initialize an empty list to store summaries
summarized_data = []

# Process each article in the data
for index, row in tqdm(data.iterrows(), total=len(data), desc="Summarizing Articles"):
    category = row['Category']
    article_text = row['Content']

    # Assign a prompt based on the category
    if category == "Trade":
        category_prompt = "Trade, with emphasis on trade policies, market dynamics, and supply chain impacts"
    elif category == "Natural Disaster":
        category_prompt = "Natural Disasters, focusing on disaster impacts on infrastructure and supply chains"
    elif category == "Geopolitics":
        category_prompt = "Geopolitics, with emphasis on international relations and policy effects on supply chains"
    elif category == "Transportation":
        category_prompt = "Transportation, focusing on logistics, shipping disruptions, and supply chain routes"
    elif category == "Suppliers":
        category_prompt = "Suppliers, focusing on sourcing, procurement, and vendor management"
    else:
        category_prompt = "General topics, with emphasis on any supply chain-related aspects"

    # Generate summary for the article
    summary = summarize_article(article_text, category_prompt)

    # Append the summary and related info to summarized_data
    summarized_data.append({
        'Category': category,
        'Article': article_text,
        'Summary': summary
    })

# Convert summarized data to a DataFrame and save as a new Excel file
output_df = pd.DataFrame(summarized_data)
output_df.to_excel('/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM Data/Final Output.xlsx', index=False)
