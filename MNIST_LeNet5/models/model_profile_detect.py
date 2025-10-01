import torch
from torch import nn
from torch.nn import Module

# LeNet5 Model definition
class LeNet5(Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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
        
    def forward (self, x):
        
        L1_zeros, L1_size, L2_zeros, L2_size, L3_zeros, L3_size, L4_zeros, L4_size, L5_zeros, L5_size = (0,) * 10 
       
        L1_zeros, L1_size = sparsity_rate(x)
        L1_smap = sparsity_map(x)
        x = self.conv1(x)
        x = self.relu1(x)    
        x = self.pool1(x)
        
        L2_zeros, L2_size = sparsity_rate(x)
        L2_smap = sparsity_map(x)
        x = self.conv2(x)
        x = self.relu2(x)      
        x = self.pool2(x)
        
        x = x.view(x.shape[0], -1)
      
        L3_zeros, L3_size = sparsity_rate(x)
        L3_smap = sparsity_map(x)
        x = self.fc1(x)
        x = self.relu3(x)

        L4_zeros, L4_size = sparsity_rate(x)
        L4_smap = sparsity_map(x)
        x = self.fc2(x)
        x = self.relu4(x)
       
        L5_zeros, L5_size = sparsity_rate(x)
        L5_smap = sparsity_map(x)
        x = self.fc3(x)
        x = self.relu5(x)
        
        zeros_list = [L1_zeros, L2_zeros, L3_zeros, L4_zeros, L5_zeros]
        sizes_list = [L1_size, L2_size, L3_size, L4_size, L5_size]
        
        # Concatenate the flattened tensors to generate a Unified Sparsity-Map
        L1_smap_flat = L1_smap.view(-1)
        L2_smap_flat = L2_smap.view(-1)
        L3_smap_flat = L3_smap.view(-1)
        L4_smap_flat = L4_smap.view(-1)
        L5_smap_flat = L5_smap.view(-1)
        smap_total = torch.cat([L1_smap_flat, L2_smap_flat, L3_smap_flat, L4_smap_flat, L5_smap_flat], dim=0)
        
        return x, zeros_list, sizes_list, smap_total

def sparsity_rate (input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return zeros, activation_count

# To compute Sparsity-Map for each layer output
def sparsity_map (input_tensor):
    
    # To create a tensor with all zeros and with the same shape of input tensor
    maps = torch.zeros_like(input_tensor, dtype=torch.uint8)
    
    # To find the positions where the input tensor is zero
    zero_positions = torch.eq(input_tensor, 0)
    
    # To set the corresponding positions in the bit_map to 1
    maps = maps.masked_fill(zero_positions, 1)
    
    return maps
