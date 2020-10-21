import torch.nn as nn


class MNIST_classifier(nn.Module):
    def __init__(self):
        super(MNIST_classifier, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(6 * 6 * 64, 128)
        self.linear_2 = nn.Linear(128, 10)
    
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x