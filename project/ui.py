from torch import nn
import torch
import pickle
import gradio as gr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sigmoid = nn.Sigmoid()

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


input_dim = 10000
hidden_dim = 64
num_layers = 4
output_dim = 1


def load_model(model_path, vectorizer_path):
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                 output_dim=output_dim, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def transform_input(input_string, vectorizer, device):
    input_vector = vectorizer.transform([input_string])
    input_vector = input_vector.toarray()
    input_vector = input_vector.reshape(
        (input_vector.shape[0], 1, input_vector.shape[-1]))
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)
    return input_tensor


# input_string1 = "Say to them,'If you don't stop plagiarizing my work, I'll have no qualms about exposing you for the fraud you are.'"
# input_string2 = "Address the issue directly with your coworker by presenting evidence of the plagiarism and seek resolution."
# input_string3 = "Say to them, 'I won't tolerate your plagiarism. Shape up or get ready for the consequences.'"
# input_string4 = "Confront them head-on about the plagiarism, and if they deny it, make sure to escalate the matter to higher-ups to ensure justice is served."
# input_string5 = "Make it crystal clear to your coworker that stealing your work won't be tolerated, and if they don't cut it out, you'll have no problem making sure everyone knows about it."

input_string1 = "Climate change is the result of human activities, primarily the burning of fossil fuels, leading to global warming and environmental disruptions."
input_string2 = "We've been treating the Earth like it's one giant garbage dump, and now we're surprised it's retaliating with freaky weather and disasters, but some people still think recycling is too much effort."
input_string3 = "Humans have been screwing over the planet so badly with their pollution and greed that now we're facing a climate crisis that could make Earth unlivable for future generations."
input_string4 = "We've been polluting the planet like there's a competition for who can wreck it the fastest, and now we're staring down the barrel of climate disaster, but some people are too busy filling their pockets to care."
input_string5 = "Listen up, we've been trashing the Earth so hard that it's like we're throwing a party on a sinking ship and nobody's even trying to plug the holes."

# input_string1 = "Any assertion of gender superiority is misguided and perpetuates harmful stereotypes."
# input_string2 = "No gender is superior, so drop the outdated superiority complex."
# input_string3 = "You're asking a dumb question. No gender is better, but if you can't grasp that, maybe you're just too thick to understand."
# input_string4 = "No gender is better; they all have their pros and cons. If you can't see that, maybe it's time to reevaluate your mindset."
# input_string5 = "The notion of one gender being better than another is quite outdated and inaccurate. All genders have their strengths and weaknesses, and it's important to appreciate and respect these differences."


def process(*inputs):

    strings = inputs
    probs = []

    model_path = "model.pth"
    vectorizer_path = "vectorizer.pkl"
    loaded_model, loaded_vectorizer = load_model(model_path, vectorizer_path)
    loaded_model.to(device)

    for input_string in strings:
        input_for_model = transform_input(input_string, loaded_vectorizer, device)
        with torch.no_grad():
            output = loaded_model(input_for_model.to(device))
        predicted_probability = output.item()
        print(f"Sentence {input_string} : Predicted probability:", predicted_probability)
        probs.append(predicted_probability)
    
    sorted_strings = [x for _, x in sorted(zip(probs, strings))]
    toxicity_dict = {}

    for i in range(len(strings)):
        toxicity_dict[strings[i]] = probs[i]

    # return_md = f'''## Least Toxic Sentence \n ### {sorted_strings[0]} \n ## Most Toxic Sentence \n ### {sorted_strings[-1]}'''
    return_md = f'''## Least Toxic Sentence \n ### {sorted_strings[0]}'''
    return return_md, toxicity_dict

iface = gr.Interface(
    fn=process,
    inputs=[
        gr.Textbox(label="Sentence 1", value=input_string1),
        gr.Textbox(label="Sentence 2", value=input_string2), 
        gr.Textbox(label="Sentence 3", value=input_string3),
        gr.Textbox(label="Sentence 4", value=input_string4),
        gr.Textbox(label="Sentence 5", value=input_string5)
    ],
    outputs=[gr.Markdown(), gr.Label(label="Toxicity Score")],
    title="CS772 DL for NLP : Course Project : Toxicity Detection for Large Language Models",
    article="Contributors : Aziz Shameem, Rohan Rajesh Kalbag, Keshav Singhal, Amruta Parulekar",
    description="Enter 5 sentences to find the least and most toxic sentences",
)

iface.launch(share=True)
