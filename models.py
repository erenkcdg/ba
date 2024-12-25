import torch.nn as nn
import torch.nn.functional as F

class MNIST_Model(nn.Module):
    def __init__(self):
        super(MNIST_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class BCW_Model(nn.Module):
    def __init__(self):
        super(BCW_Model, self).__init__()
        self.layer_1 = nn.Linear(30, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 2)
        
        self.relu = nn.ReLU()
        self.layernorm1 = nn.LayerNorm(64)
        self.layernorm2 = nn.LayerNorm(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.layernorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.layernorm2(x)
        x = self.layer_out(x)
        return x
