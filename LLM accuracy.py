import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def load_training_data(file_path):
    df = pd.read_excel(file_path)  # Assuming it's an Excel file
    df = df[['summary', 'category']]  # Ensure the correct columns are selected
    df.dropna(inplace=True)  # Drop any rows with missing data
    return df
def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    return df, label_encoder
def train_model(train_data, val_data, tokenizer, label_encoder, model_name='bert-base-uncased'):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    model = model.to(device)
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    def tokenize_function(example):
        return tokenizer(example['summary'], padding="max_length", truncation=True)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')
    return model
def classify_summaries(model, tokenizer, input_file, output_file, label_encoder):
    # Load the new data
    df = pd.read_excel(input_file)
    df['summary'] = df['summary'].fillna('')
    tokenized_data = tokenizer(df['summary'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokenized_data = {k: v.to(device) for k, v in tokenized_data.items()}
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(input_ids=tokenized_data['input_ids'], attention_mask=tokenized_data['attention_mask'])
        predictions = torch.argmax(outputs.logits, dim=-1)
    predictions = predictions.cpu()
    df['Predicted Category'] = label_encoder.inverse_transform(predictions.numpy())
    output_df = df[['summary', 'Predicted Category']]
    output_df.to_excel(output_file, index=False)
    print(f"The categorized file has been saved and is available at: {output_file}")
if __name__ == "__main__":
    training_file_path = "/Users/sathishm/Documents/TSM Folder/Training Data/Training data.xlsx"
    new_summaries_file_path = "/Users/sathishm/Documents/TSM Folder/Scrapping Output data/Test_data_output.xlsx"
    output_file_path = os.path.join(os.getcwd(), "/Users/sathishm/Documents/TSM Folder/LLM Output data/Output.xlsx")
    df = load_training_data(training_file_path)
    df, label_encoder = preprocess_data(df)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = train_model(train_df, val_df, tokenizer, label_encoder)
    classify_summaries(model, tokenizer, new_summaries_file_path, output_file_path, label_encoder)