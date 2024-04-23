'''architecture  for neural networks used in the task'''
from torch import nn

class ThermoNet(nn.Module):
    '''class for simple neural network for predicting thermostability from one-hot encoded protein
        note: designed for use of specific shape of data as input, might need changing if 
    '''
    def __init__(self, max_seq_length=466, no_unique_aa=21):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(max_seq_length * no_unique_aa, 512),
            nn.Dropout1d(0.1),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout1d(0.1), #dropout to avoid overfitting
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits