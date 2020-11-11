import random
import json
import torch
from model import NeuralNet
from utilities import sack_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

botname = "Zach"
print('Meet our chatbot! Type \"quit\" to exit the chat')
while True:
    sentence = input("You: ")
    if sentence == 'quit':
        break
    '''We want to tokenize the setence and then calculate
    a sack of words, like we did for our training data '''
    sentence = tokenize(sentence)
    x = sack_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    #sack_of_words returns a numpy array
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    #actual tag,    tag label aka the number        
    tag = tags[predicted.item()]

    '''We can check if the probability for the tag is high enough. This is one way
    we can improve this program. Let's say 75 percent accuracy is good enough, otherwise we can
    say that the bot simply didn't understand what the user said. '''
    '''We want to find the corresponding intents, so we'll have to
    loop through all of the intents json and see if the tag matches'''

    probability = torch.softmax(output, dim=1)
    prob = probability[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{botname}: {random.choice(intent['responses'])}")
    else:
        print(f'{botname}: I\'m sorry, I could not understand what you were saying....')