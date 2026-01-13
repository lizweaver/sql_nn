import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    """Simple neural network processing flattened MNIST images
    with 128 neurons in the first hidden layer, 64 in the second hidden layer,
    and 10 in the output layer"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            return output.numpy()
        



