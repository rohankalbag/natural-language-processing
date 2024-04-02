## CS772 : Deep Learning for NLP : Assignment 2

This is our attempt at the second assignment of the course CS772.  

First create a python virtual environment and then install all dependencies using

```bash
pip install -r requirements.txt
```

The learned weights of the final model are loaded in the form of a pickled in the UI for inference. They can be found in `model_3000_10.pkl`

To boot up the User Interface for model inference    

```bash
python3 ui.py
```

A project by:-

1) Aziz Shameem : 20d070020  
2) Rohan Rajesh Kalbag : 20d170033  
3) Amruta Parulekar : 20d070009  
4) Keshav Singhal : 20d070047

PDF of slides has been added in the folder as cs772-2024-assignment2.pdf.

Dataloader_example.ipynb contains code to load data, train model and test it, including 5 fold cross validation.

utils.py contains the code for the recuurent perceptron along with forward pass, back propagation with time implementation and the training loop.

ui.py contains the code for the UI.

pos_tag.ipynb contains the code to use the NLTK toolkit to generate POS tags for a sentence input to the UI.

Analysis.ipynb contains the printed error cases and their analysis (conclusions in PPT).

Final trained model weights with best accuracy are in model_3000_10.pkl.

