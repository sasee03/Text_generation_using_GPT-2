# GPT-2 Text Generation App

This repository contains a simple text generation app built using GPT-2 and Streamlit. The app allows users to input text and generate new text based on GPT-2's language model. It consists of two main files: `main.py` for handling model loading and text generation, and `app.py` for the Streamlit-based user interface.

## Features

- **Text Generation**: Users can input text and generate continuations using GPT-2.
- **Adjustable Parameters**: Users can adjust the maximum length of generated text and control the "creativity" (temperature) of the output.
- **Background Image**: The app includes a customizable background image using a local file.

## Project Structure

```
|-- app.py                  # Streamlit app for user interface
|-- main.py                 # Core logic for model loading and text generation
|-- img.jpg                 # Background image for the Streamlit app
|-- README.md               # Project documentation (this file)
|-- requirements.txt        # Python dependencies
|-- gpt2/                   # Directory containing GPT-2 model files (unzipped)
```

## Setup Instructions

### 1. Install Dependencies

Before running the app, install the necessary Python packages. All required packages are listed in `requirements.txt`. You can install them using the following command:

```bash
pip install -r requirements.txt
```

### 2. Download and Prepare GPT-2 Model

To use GPT-2 locally, you must download the pre-trained model. If the model is in a zip file, place it in the project directory and unzip it using the function in `main.py`.

- Unzip the model:
    - Download the GPT-2 model files (if needed) from [Hugging Face Model Hub](https://huggingface.co/gpt2).
    - Unzip the model and store the contents in a folder called `gpt2` inside the project directory.

### 3. Running the App

To start the Streamlit app, run the following command in your terminal:

```bash
streamlit run app.py
```

The app should open in your browser. You can input some text, adjust the settings, and generate text using GPT-2.

### 4. Customizing the Background Image

The app allows you to set a background image by replacing the `img.jpg` file with your own image. Simply provide the path to the image in the `add_bg_from_local()` function in `app.py`.

## File Descriptions

- **`main.py`**:
    - Contains functions to unzip the GPT-2 model, load the model and tokenizer, and generate text based on user input.
  
- **`app.py`**:
    - Handles the Streamlit frontend, including loading the model, getting user input, and displaying the generated text.

- **`requirements.txt`**:
    - Lists all the dependencies required for the project (e.g., `transformers`, `torch`, `streamlit`).

- **`gpt2/`**:
    - Directory for storing the unzipped GPT-2 model files. This folder should contain model weights and configuration files after extraction.

## Example Usage

1. Run the app using the terminal.
2. In the browser, input a sentence or phrase to begin generating text.
3. Adjust the parameters like the maximum text length or creativity (temperature) using sliders.
4. Click "Generate Text" and view the generated output.

## Dependencies

- `transformers`: For loading the GPT-2 model and tokenizer.
- `torch`: For handling model computations on CPU/GPU.
- `streamlit`: For building the user interface.

You can install the dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.

---

## Credits

- [Hugging Face](https://huggingface.co) for the pre-trained GPT-2 model.
- [Streamlit](https://streamlit.io) for the app framework.
