import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import zipfile

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to unzip the downloaded model into a folder if it's not already extracted
def unzip_model(zip_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Model unzipped to {extract_path}")
    else:
        print(f"Model already exists in {extract_path}")

# Function to load GPT-2 model and tokenizer from the extracted folder
def load_gpt2_model(local_model_path):
    model_tf = GPT2LMHeadModel.from_pretrained(local_model_path).to(device)
    tokenizer_tf = GPT2Tokenizer.from_pretrained(local_model_path)
    return model_tf, tokenizer_tf

# Function to generate text using the model and tokenizer
def generate_text(model_tf, tokenizer_tf, input_text, max_length=100, temperature=1.0):
    inputs = tokenizer_tf.encode(input_text, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        outputs = model_tf.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_p=0.9,
            do_sample=True
        )

    # Decode and return the generated text
    generated_text = tokenizer_tf.decode(outputs[0], skip_special_tokens=True)
    return generated_text
