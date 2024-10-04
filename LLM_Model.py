import pandas as pd
import torch
import os
import numpy as np
import pickle  # Import pickle to save the LabelEncoder
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set device (MPS, GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_training_data(file_path):
    """
    Load training data from Excel.
    """
    df = pd.read_excel(file_path)
    df = df[['summary', 'category']]
    df.dropna(inplace=True)
    return df


def preprocess_data(df):
    """
    Encode category labels and return the DataFrame with labels and the LabelEncoder.
    """
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    return df, label_encoder


def check_class_distribution(df):
    """
    Check the distribution of classes and return the class weights if needed.
    """
    class_counts = df['label'].value_counts().sort_index().values
    print("Class Distribution:", class_counts)

    # Compute class weights based on inverse frequency of the classes
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    return class_weights


def compute_metrics(eval_pred):
    """
    Compute metrics such as accuracy, precision, recall, and F1-score.
    """
    logits, labels = eval_pred
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    predictions = torch.argmax(logits, dim=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def tokenize_function(tokenizer, example):
    """
    Tokenize the input data using the tokenizer with reduced max length for faster training.
    """
    return tokenizer(example['summary'], padding="max_length", truncation=True, max_length=128)  # Reduced max_length


class WeightedTrainer(Trainer):
    """
    Custom Trainer class that applies class weights to the loss function.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Use weighted cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_model(train_data, val_data, tokenizer, label_encoder, class_weights, model_name='bert-base-uncased'):
    """
    Train the model with class weights to handle imbalanced data.
    """
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    model = model.to(device)

    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    train_dataset = train_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=1e-5,  # Adjusted learning rate for fine-tuning
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,  # Increased to 5 epochs
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        gradient_accumulation_steps=2  # Simulate larger batch size
    )

    # Disable fp16 if using MPS (Apple Silicon)
    if device.type == "mps":
        training_args.fp16 = False  # Disable fp16 if using MPS

    # Use the custom WeightedTrainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights  # Pass class weights here
    )

    trainer.train()

    # Save the model, tokenizer, and label encoder
    model.save_pretrained("/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM")
    tokenizer.save_pretrained("/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM")

    # Save the label encoder
    with open("/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM/label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)

    print(
        "Model, tokenizer, and label encoder have been saved to '/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM/'.")


if __name__ == "__main__":
    training_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Training Data/Training data.xlsx"

    # Load and preprocess the data
    df = load_training_data(training_file_path)
    df, label_encoder = preprocess_data(df)

    # Check class distribution and compute class weights
    class_weights = check_class_distribution(df)

    # Split into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Train the model with class weights to handle imbalanced data
    train_model(train_df, val_df, tokenizer, label_encoder, class_weights)