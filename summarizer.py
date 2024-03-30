import gradio as gr
from transformers import BartForConditionalGeneration, BartTokenizer
from PyPDF2 import PdfFileReader
import torch

# Loading BART 
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

# Function to Calculate summary lengths 
def calc_summary_lengths(text_length):
    short_min = int(0.10 * text_length)
    medium_min = short_max_ =  int(0.15 * text_length)
    medium_max = long_min =  int(0.20 * text_length)
    long_max = int(0.30 * text_length)
    return {
        "Short": (short_min, short_max_),
        "Medium": (medium_min, medium_max),
        "Long": (long_min, long_max)
    }

# Function to summarize text
def summarize_text(pdf_file, summary_length):
    try:
        text = ""
        with open(pdf_file.name, "rb") as f:
            reader = PdfFileReader(f)
            for page in range(reader.numPages):
                text += reader.getPage(page).extractText()

        text = " ".join(text.split())
        text_length = len(text.split())

        summary_range = calc_summary_lengths(text_length)
        min_length, max_length = summary_range[summary_length]

        # Summary Generation
        inputs = tokenizer.encode(text, max_length=1024, return_tensors='pt', truncation=True).to(device)
        summary_ids = model.generate(inputs, num_beams=4, min_length=min_length, max_length=max_length, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    except Exception as e:
        return f"Error: {str(e)} \nPlease check file size and type!"


#Gradio Interface
input_component = gr.File(label="Upload PDF file")
output_component = gr.Textbox(label="Summarized Text")
summary_length_component = gr.Dropdown(label="Summary Length", choices=["Short", "Medium", "Long"])

title = "PDF Text Summarizer (BART)"
description = "<h2>Upload a PDF file and select the desired summary length.</h2>"

InterFace = gr.Interface(fn=summarize_text, inputs=[input_component, summary_length_component], outputs=output_component, title=title, description=description)
InterFace.launch()
