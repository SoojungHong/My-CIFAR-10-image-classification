# Imports
# ---------------
from datetime import datetime, timedelta
from tensorflow.python.ops.nn_ops import dropout

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# Data
# ---------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

max_epochs = 10
batch_size = 1000

train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
train_count = len(train_ds)

test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_ds_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
test_count = len(test_ds)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Prepare
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
    [2048, 1024],
    [1024, 512],
    [512, 256],
    [256, 128],
    [128, 64],
    [64, output_classes]
]

# Model
# --------------
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


# Loss
# -------------
def criterion():
    return nn.CrossEntropyLoss()

# Optimizer
# -------------
def optimizer_fn(model):
    learning_rate = 1e-6
    return optim.Adam(model.parameters(), lr=learning_rate)

def training():
    # Training
    # -------------
    model = Model()
    optimizer = optimizer_fn(model)
    loss_fn = criterion()
    start_time = datetime.now()
    for e in range(max_epochs):
        steps = e*train_count
        running_loss = 0.0
        for steps, data, in enumerate(train_ds_loader, 0):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % 200 == 0:
                print("Esapsed Time : {}s  >  Steps : {}  >  Loss : {}".format(timedelta(seconds=(datetime.now() - start_time).total_seconds()), steps, running_loss/2000))

        # Evaluation
        correct = 0
        total = 0
        with torch.no_grad():
            for steps, data, in enumerate(test_ds_loader):
                images, labels = data
                pred = model(images)
            
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy : {:.0%}".format(correct/total))

if __name__=='__main__':
    training()