import gradio as gr
import re
import numpy as np
from utils import RecurrentPerceptron
from utils import DataLoader

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

pos_mapping = {
    "NN": 1,
    "DT": 2,
    "JJ": 3,
    "OT": 4
}

def convert_pos_tag(tag):
    return pos_mapping.get(tag, pos_mapping["OT"])

def generate_pos_tags(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    formatted_tags = [(word, convert_pos_tag(tag)) for word, tag in tagged_words]
    return formatted_tags, words

def evaluate(sentence):
    model = RecurrentPerceptron()
    model.load_model('model.pkl')
    W = [float(k) for k in model.W]
    V = model.V

    pos_tags, tokens = generate_pos_tags(sentence)
    sentence = [word for word, tag in pos_tags]
    pos_tags = [tag for word, tag in pos_tags]
    data = [{"tokens": tokens, "pos_tags": pos_tags}]
    
    data_loader = DataLoader(data, test=True)
    transformed = next(data_loader)
    pred = model.infer(transformed[0]["pos_tags"])
    pred = [int(m) for m in pred]
    return (W, V, tokens, pos_tags, pred)

def process(input_text):
    W, V, tokens, pos_tags, pred = evaluate(input_text)

    return f"### Weights W : \n\t{W}\n\n ### Weight V : \n\t{V}\n\n ### Tokens : \n\t`{tokens}`\n\n ### NLTK Predicted POS Tags : \n\t`{pos_tags}`\n\n ### Predicted Chunks : \n\t`{pred}`"

iface = gr.Interface(
    fn=process,
    inputs=gr.Textbox(),
    outputs=gr.Markdown(),
    title="CS772 DL for NLP : Assignment 2 : POS Chunking using Recurrent Perceptron", 
    article="Contributors : Aziz Shameem, Rohan Rajesh Kalbag, Keshav Singhal, Amruta Parulekar", 
    description="Enter a sentence to get the POS tags and chunks.", 
)

iface.launch()
