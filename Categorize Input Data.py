import os
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import logging
import traceback
from functools import wraps

# Configure logging for complex debugging information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Set device (CPU or MPS/GPU) dynamically
def get_device():
    """Dynamically determine the device (CPU, MPS, GPU) based on availability."""
    if torch.backends.mps.is_available():
        logging.info("Using MPS (Apple Silicon).")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logging.info("Using CUDA (GPU).")
        return torch.device("cuda")
    else:
        logging.info("Using CPU.")
        return torch.device("cpu")


device = get_device()


# Decorator to handle exceptions and log stack traces
def exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in function '{func.__name__}': {e}")
            logging.error(traceback.format_exc())
            raise e

    return wrapper


@exception_handler
def load_model_and_tokenizer(model_dir):
    """Load the pre-trained model and tokenizer from the specified directory with deep validation."""
    if not os.path.exists(model_dir):
        logging.error(f"Model directory '{model_dir}' does not exist. Please provide a valid path.")
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

    logging.info(f"Loading model from '{model_dir}'...")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    logging.info("Model and tokenizer successfully loaded.")
    return model.to(device), tokenizer


@exception_handler
def load_data(input_file):
    """Load the input Excel file containing summaries and validate its content."""
    if not os.path.exists(input_file):
        logging.error(f"Input file '{input_file}' does not exist.")
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

    logging.info(f"Loading data from '{input_file}'...")
    df = pd.read_excel(input_file)

    if 'summary' not in df.columns:
        logging.error("The 'summary' column is missing from the input file.")
        raise ValueError("Input file must contain a 'summary' column.")

    df['summary'] = df['summary'].fillna('')  # Handle missing summaries with logging
    if df['summary'].empty:
        logging.error("No valid summaries found in the input file.")
        raise ValueError("No valid summaries found in the input file.")

    logging.info(f"Data from '{input_file}' successfully loaded and validated.")
    return df


@exception_handler
def tokenize_data(tokenizer, df):
    """Tokenize the summaries using the pre-trained tokenizer with exception handling."""
    logging.debug(f"Tokenizing {len(df)} summaries...")
    tokenized_data = tokenizer(
        df['summary'].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    logging.debug(f"Tokenization complete. Moving data to device: {device}.")
    return {k: v.to(device) for k, v in tokenized_data.items()}


@exception_handler
def infer_categories(model, tokenized_data):
    """Perform inference to predict class labels and handle exceptions."""
    logging.debug("Running inference on tokenized summaries...")
    model.eval()

    with torch.no_grad():
        try:
            outputs = model(
                input_ids=tokenized_data['input_ids'],
                attention_mask=tokenized_data['attention_mask']
            )
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        except Exception as e:
            logging.error("Error during model inference.")
            raise RuntimeError(f"Error during model inference: {e}")

    predictions = np.array(predictions, dtype=int).flatten()  # Ensure it's a 1D numpy array
    logging.debug(f"Inference complete. Predictions shape: {predictions.shape}, type: {predictions.dtype}")
    return predictions


@exception_handler
def validate_predictions(predictions, label_encoder):
    """Ensure predictions are valid class indices and provide extensive logging."""
    num_classes = len(label_encoder.classes_)
    logging.debug(f"Validating predictions against {num_classes} classes...")

    if predictions.max() >= num_classes or predictions.min() < 0:
        logging.error(f"Invalid prediction indices: {predictions}.")
        raise ValueError(f"Prediction values out of range. Valid indices: [0, {num_classes - 1}]")

    logging.debug("Predictions validated successfully.")
    return predictions


@exception_handler
def classify_summaries(model, tokenizer, input_file, output_file, label_encoder):
    """Classify summaries and save the categorized output with extensive logging."""
    df = load_data(input_file)
    tokenized_data = tokenize_data(tokenizer, df)
    predictions = infer_categories(model, tokenized_data)
    predictions = validate_predictions(predictions, label_encoder)

    try:
        df['Predicted Category'] = label_encoder.inverse_transform(predictions)
    except ValueError as e:
        logging.error(f"Error during label encoding: {e}. Assigning 'Unknown' category.")
        df['Predicted Category'] = 'Unknown'

    df[['summary', 'Predicted Category']].to_excel(output_file, index=False)
    logging.info(f"Results saved to {output_file}")


@exception_handler
def load_label_encoder(label_encoder_path):
    """Load the label encoder from a pickle file with additional validation."""
    if not os.path.exists(label_encoder_path):
        logging.error(f"Label encoder file '{label_encoder_path}' does not exist.")
        raise FileNotFoundError(f"Label encoder file '{label_encoder_path}' does not exist.")

    logging.info(f"Loading label encoder from '{label_encoder_path}'...")
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    logging.info("Label encoder loaded successfully.")
    return label_encoder


if __name__ == "__main__":
    # File paths
    input_file = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Scrapping Output data/Data_syndicate.xlsx"
    output_file = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM Output data/Datathon.xlsx"
    model_dir = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM"
    label_encoder_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM/label_encoder.pkl"

    # Load resources
    label_encoder = load_label_encoder(label_encoder_path)
    model, tokenizer = load_model_and_tokenizer(model_dir)

    # Perform classification with exception handling
    try:
        classify_summaries(model, tokenizer, input_file, output_file, label_encoder)
    except Exception as e:
        logging.error(f"Classification failed: {e}")