import torch
from torch import nn
from torch.nn import Module

# LeNet5 Model definition
class LeNet5(Module):
    def __init__(self, args):
        super(LeNet5, self).__init__()
        self.beta  = args.beta
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3   = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        
        L1_zeros, L1_size, L2_zeros, L2_size, L3_zeros, L3_size, L4_zeros, L4_size, L5_zeros, L5_size = (0,) * 10
        
        L1_zeros, L1_size = sparsity_rate(x)
        L1_elements = x[0].numel()
        L1_tanh = tanh(x, self.beta) # Estimates the number of non-zero elements for each image in the batch
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        L2_zeros, L2_size = sparsity_rate(x)
        L2_elements = x[0].numel()
        L2_tanh = tanh(x, self.beta)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.shape[0], -1)

        L3_zeros, L3_size = sparsity_rate(x)
        L3_elements = x[0].numel()
        L3_tanh = tanh(x, self.beta)
        x = self.fc1(x)
        x = self.relu3(x)

        L4_zeros, L4_size = sparsity_rate(x)
        L4_elements = x[0].numel()
        L4_tanh = tanh(x, self.beta)
        x = self.fc2(x)
        x = self.relu4(x)

        L5_zeros, L5_size = sparsity_rate(x)
        L5_elements = x[0].numel()
        L5_tanh = tanh(x, self.beta)
        x = self.fc3(x)
        x = self.relu5(x)

        # We add all tanh outputs which represent the number of non-zeros in each image of the batch
        # Then, we divide them by number_of_network_neurons to estimate non-zero rate for each image
        tanh_total = (L1_tanh + L2_tanh + L3_tanh + L4_tanh + L5_tanh)
        tanh_total = (tanh_total)/(L1_elements + L2_elements + L3_elements + L4_elements + L5_elements)
        
        zeros_list = [L1_zeros, L2_zeros, L3_zeros, L4_zeros, L5_zeros]
        sizes_list = [L1_size, L2_size, L3_size, L4_size, L5_size]

        return x, zeros_list, sizes_list, tanh_total

# To estimate number of non-zero elements for each image in the batch using Tanh
def tanh(input_tensor, beta):
    # To scale the tensor by BETA and apply tanh function to the scaled tensor
    output = torch.tanh(beta * input_tensor)
    # To sum the activations separately for each image in the batch
    output = output.view(input_tensor.size(0), -1).sum(dim=1)
    return output

def sparsity_rate(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return zeros, activation_count
