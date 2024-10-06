import os
import pickle
import logging
import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device (MPS, GPU, or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_training_data(file_path):
    """Load training data from an Excel file, ensure the presence of necessary columns."""
    logging.info(f"Loading training data from: {file_path}")
    df = pd.read_excel(file_path)
    if 'summary' not in df or 'category' not in df:
        raise ValueError("Input file must contain 'summary' and 'category' columns.")
    df = df[['summary', 'category']].dropna()
    return df


def preprocess_data(df):
    """Encode category labels and return the DataFrame with labels and the LabelEncoder."""
    logging.info("Preprocessing data and encoding labels.")
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    return df, label_encoder


def tokenize_function(tokenizer, texts, labels):
    """Tokenize the input text data for BERT and include labels."""
    logging.info(f"Tokenizing {len(texts)} text entries using BERT tokenizer.")
    # Convert labels to a list or array to avoid pandas index issues
    labels = torch.tensor(labels.tolist()).to(device)

    tokenized_data = tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    tokenized_data['labels'] = labels
    return tokenized_data


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1-score for evaluation."""
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def train_model(train_data, val_data, tokenizer, label_encoder, class_weights, model_name='bert-base-uncased'):
    """Train the BERT model with the tokenized data and class weights."""
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    model.to(device)

    # Prepare datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_data['input_ids'],
        'attention_mask': train_data['attention_mask'],
        'labels': train_data['labels']
    })
    val_dataset = Dataset.from_dict({
        'input_ids': val_data['input_ids'],
        'attention_mask': val_data['attention_mask'],
        'labels': val_data['labels']
    })

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        gradient_accumulation_steps=2
    )

    if device.type == "mps":
        training_args.fp16 = False  # Disable fp16 for MPS

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting training...")
    trainer.train()

    # Save the model, tokenizer, and label encoder
    output_dir = "./saved_model"
    logging.info(f"Saving model and tokenizer to {output_dir}.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(f"{output_dir}/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)

    logging.info("Training complete and models saved successfully.")


if __name__ == "__main__":
    training_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Training Data/Training data.xlsx"

    # Load and preprocess the data
    df = load_training_data(training_file_path)
    df, label_encoder = preprocess_data(df)

    # Split data into training and validation sets and reset indices
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the data with labels
    tokenized_train = tokenize_function(tokenizer, train_df['summary'], train_df['label'])
    tokenized_val = tokenize_function(tokenizer, val_df['summary'], val_df['label'])

    # Compute class weights (optional, modify as per your dataset's class distribution)
    class_weights = torch.ones(len(label_encoder.classes_)).to(device)

    # Train the model
    train_model(tokenized_train, tokenized_val, tokenizer, label_encoder, class_weights)
