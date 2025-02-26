import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Fully connected layer

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))  # Flatten the input tensor to 1D

