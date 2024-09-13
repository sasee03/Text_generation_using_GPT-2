import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import base64

# Function to add background image from a local path
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with the path to your local image
add_bg_from_local(r"img.jpg")  # Use your image path

# Load the GPT-2 model and tokenizer
@st.cache_resource
def load_gpt2():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

model, tokenizer = load_gpt2()

# Streamlit app title
st.title("Interactive GPT-2 Text Generation")

# Input text box
input_text = st.text_input("Enter some text to start generating:", "")

# Parameters for text generation
max_length = st.slider("Max length of generated text:", 50, 300, 100)
temperature = st.slider("Temperature (creativity level):", 0.7, 1.5, 1.0, step=0.1)

# Generate button
if st.button("Generate Text"):
    if input_text.strip():
        # Tokenize input text
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_p=0.9,
                do_sample=True
            )

        # Decode and display the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Generated Text:")
        st.write(generated_text)
    else:
        st.write("Please enter some text to generate.")
