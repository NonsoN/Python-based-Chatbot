import json
import numpy as np
# noinspection PyUnresolvedReferences
from model import NeuralNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# noinspection PyUnresolvedReferences
from utilities import tokenize, stem, sack_of_words
with open("intents.json", 'r') as file:
    intents = json.load(file)

#An array holding all of the possible words
all_words = []
#Array to hold the tags of each word
tags = []
#Array to hold both our patterns and tags
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w) #extend because we're not creating an array of arrays
        xy.append((w, tag))

ignore_words = ['?', '!', ',', '.', ':', ';']
#Simply put, all words will contain the stem of all the words in the original array
#but also exclude whatever is in ignore words
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Converting it to a set, allows us to avoid repetiton
'''all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(all_words)'''

'''CREATING THE TRAINING DATA'''

#all the bag_of_words
x_trainer = []
#associated number for each tag
y_trainer = []

#We'll need to unpack the tuple since that is what xy contains
for (pattern_sent, tag) in xy:
    sack = sack_of_words(pattern_sent, all_words)
    x_trainer.append(sack)

    label = tags.index(tag)
    y_trainer.append((label))
x_trainer = np.array(x_trainer)
y_trainer = np.array(y_trainer)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(x_trainer)
        self.x_data = x_trainer
        self.y_data = y_trainer

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


batch_size = 8
hidden_size = 8
#number of different tags that we have
output_size = len(tags)
#length of each sack of words that we've created. All sacks have the same size
input_size = len(x_trainer[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
trainer_loader = DataLoader(dataset=dataset, batch_size=batch_size)


dataset = ChatDataset()
trainer_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle = True, num_workers = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in trainer_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}
FILE = 'data.pth'
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')