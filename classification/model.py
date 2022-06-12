
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Given Model
# --------------

dropout_rate = 0.5
output_classes = len(classes)
weights = [
    [3, 32, 3],
    [32, 32, 3],
    [32, 64, 3],
    [64, 64, 3],
    [64, 128, 3],
    [128, 256, 3],
    [4096, 1024],  
    [1024, 512],
    [512, 256],
    [256, 128],
    [128, 64],
    [64, output_classes]
]

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = self.conv2d(weights[0])
        self.conv2 = self.conv2d(weights[1])
        self.conv3 = self.conv2d(weights[2])
        self.conv4 = self.conv2d(weights[3])
        self.conv5 = self.conv2d(weights[4])
        self.conv6 = self.conv2d(weights[5])

        self.pool = nn.MaxPool2d(2, 2)

        self.d1 = self.dense(weights[6])
        self.d2 = self.dense(weights[7])
        self.d3 = self.dense(weights[8])
        self.d4 = self.dense(weights[9])
        self.d5 = self.dense(weights[10])
        self.d6 = self.dense(weights[11])

    def conv2d(self, filters, padding='same'):
        return nn.Conv2d(*filters, padding=padding)

    def dense(self, features):
        return nn.Linear(*features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        flatten = torch.flatten(x, 1)

        x = F.dropout(F.relu(self.d1(flatten)), p=dropout_rate)
        x = F.dropout(F.relu(self.d2(x)), p=dropout_rate)
        x = F.dropout(F.relu(self.d3(x)), p=dropout_rate)
        x = F.dropout(F.relu(self.d4(x)), p=dropout_rate)
        x = F.dropout(F.relu(self.d5(x)), p=dropout_rate)

        logits = self.d6(x)

        return logits


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x