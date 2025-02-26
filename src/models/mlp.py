import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_classes, input_dim=32 * 32 * 3, hidden_units=[512, 256], dropout_rate=0.5):
        super(MLP, self).__init__()

        # Create a list of fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = input_dim

        for hidden_size in hidden_units:
            self.fc_layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size  # Output of current layer is input to next layer

        # Output layer
        self.fc_out = nn.Linear(in_features, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.size(0), -1)  # Flatten the input image (batch_size, input_dim)

        # Apply each fully connected layer with ReLU and dropout
        for fc in self.fc_layers:
            x = F.relu(fc(x))  # Apply ReLU activation
            x = self.dropout(x)  # Apply dropout

        # Final output layer
        x = self.fc_out(x)

        return x
