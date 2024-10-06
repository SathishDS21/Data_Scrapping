import pandas as pd
import torch
import os
import numpy as np
import pickle
from datasets import Dataset
from transformers import XLNetTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up device for GPU/CPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_training_data(file_path):
    logging.info(f"Loading training data from: {file_path}")
    df = pd.read_excel(file_path)
    if 'summary' not in df or 'category' not in df:
        raise ValueError("Input file must contain 'summary' and 'category' columns.")
    df = df[['summary', 'category']].dropna()
    logging.info(f"Loaded {len(df)} rows.")
    return df

def preprocess_data(df):
    logging.info("Preprocessing data and encoding labels.")
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    logging.info(f"Classes found: {list(label_encoder.classes_)}")
    return df, label_encoder

def tokenize_function(tokenizer, texts, labels):
    logging.info("Tokenizing text data using XLNet tokenizer.")
    tokenized_data = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    tokenized_data = {key: value.cpu().numpy() for key, value in tokenized_data.items()}
    tokenized_data['labels'] = np.array(labels)
    return tokenized_data

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def train_model(train_dataset, val_dataset, tokenizer, label_encoder, class_weights, model_name='xlnet-base-cased'):
    # Load XLNet model for sequence classification
    model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    model.to(device)

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
        training_args.fp16 = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting training...")
    trainer.train()

    output_dir = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM"
    logging.info(f"Saving model and tokenizer to {output_dir}.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(f"{output_dir}/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)

    logging.info("Training complete and models saved successfully.")

if __name__ == "__main__":
    training_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Training Data/Training data.xlsx"

    df = load_training_data(training_file_path)
    df, label_encoder = preprocess_data(df)

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Initialize XLNet tokenizer
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    # Tokenize the training and validation data
    tokenized_train = tokenize_function(tokenizer, train_df['summary'], train_df['label'])
    tokenized_val = tokenize_function(tokenizer, val_df['summary'], val_df['label'])

    # Convert tokenized data into HuggingFace-style datasets
    train_dataset = Dataset.from_dict(tokenized_train)
    val_dataset = Dataset.from_dict(tokenized_val)

    class_weights = torch.ones(len(label_encoder.classes_)).to(device)

    # Train the model
    train_model(train_dataset, val_dataset, tokenizer, label_encoder, class_weights)
