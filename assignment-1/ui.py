import gradio as gr
import re

def process(input_text):
    pattern = r'^[01]{10}$'
    if re.match(pattern, input_text):
        return check_if_palindrome(input_text)
    else:
        return "Invalid input. Please enter a 10-bit binary number."

def check_if_palindrome(input):
    return "Is a Palindrome" if input == input[::-1] else "Not a Palindrome"

iface = gr.Interface(
    fn=process,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    title="CS772 DL for NLP : Assignment 1 : Palindrome Checker using Multilayer Perceptron", 
    article="Contributors : Aziz Shameem, Rohan Rajesh Kalbag, Keshav Singhal, Amruta Parulekar", 
    description="Enter 10 bit number to check if it is a palindrome or not.", 
)

iface.launch()
