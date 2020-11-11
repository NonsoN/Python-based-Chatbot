import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    '''Feed forward neural nets gets our bag of words as an input.
    One fully connected layer has the number of patterns and hidden layers as input size
    The output size is the number of different classes.
    Input size and number of classes are fixed. hidden size is not'''
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size) 
        self.layer2 = nn.Linear(hidden_size, hidden_size) 
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)

        return out
