import gradio as gr
import re
import numpy as np
import pickle

path = 'model_wts.pkl'

with open(path, 'rb') as file :
    MODEL_WEIGHTS = pickle.load(file)

def evaluate(weights, inp, threshold=0.9) :
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    weights = np.array(weights)
    inp = np.array(inp)
    one = sum(weights[:10]*inp) + weights[10]
    two = sum(weights[11:21]*inp) + weights[21]
    three = sum(weights[22:32]*inp) + weights[32]
    four = sum(weights[33:43]*inp) + weights[43]
    vals = np.array([one, two, three, four])
    vals = sigmoid(vals)
    
    return 1 if sigmoid(sum(weights[44:48]*vals) + weights[48]) > threshold else 0

def process(input_text):
    pattern = r'^[01]{10}$'
    if re.match(pattern, input_text):
        bitlist = [int(k) for k in list(input_text)]
        return check_if_palindrome(MODEL_WEIGHTS, bitlist)
    else:
        return "Invalid input. Please enter a 10-bit binary number."

def check_if_palindrome(weights, input):
    return "Is a Palindrome" if evaluate(weights, input) else "Not a Palindrome"

iface = gr.Interface(
    fn=process,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    title="CS772 DL for NLP : Assignment 1 : Palindrome Checker using Multilayer Perceptron", 
    article="Contributors : Aziz Shameem, Rohan Rajesh Kalbag, Keshav Singhal, Amruta Parulekar", 
    description="Enter 10 bit number to check if it is a palindrome or not.", 
)

iface.launch()
