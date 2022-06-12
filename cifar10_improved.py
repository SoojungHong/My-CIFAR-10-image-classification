# Imports
# ---------------
from datetime import datetime, timedelta
# FIX from tensorflow.python.ops.nn_ops import dropout

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import math

# Data
# ---------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

max_epochs = 10
batch_size = 4  # 1000

train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
train_count = len(train_ds)

test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_ds_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
test_count = len(test_ds)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if not torch.cuda.is_available():
    raise Exception('At least one gpu must be available')
gpu = torch.device('cuda')


# Model
# --------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)  # input channel, output_channel, kernel_size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Model with Batch normalization
class BN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6) # Batch Normalization
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


# Loss
# -------------
def criterion():
    return nn.CrossEntropyLoss()


# Optimizer
# -------------
def adam_optimizer_fn(model):
    learning_rate = 1e-6
    return optim.Adam(model.parameters(), lr=learning_rate)


def sgd_optimizer_fn(model):
    learning_rate = 0.001
    return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# Training
# -------------
def training(opti, bn):

    # select Model
    if bn == True:
        model = BN_Model()
    else:
        model = Model()

    model.to(gpu)

    # select optimizer
    if opti == 'SGD':
        optimizer = sgd_optimizer_fn(model)
    elif opti == 'ADAM':
        optimizer = adam_optimizer_fn(model)

    loss_fn = criterion()
    start_time = datetime.now()
  
    for e in range(max_epochs):
        print('epoch : ', e)
        steps = e * train_count
        running_loss = 0.0
        for steps, data, in enumerate(train_ds_loader, 0):
            images, labels = data[0].to(gpu), data[1].to(gpu)  # use gpu
            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % 200 == 0:
                print("Esapsed Time : {}s  >  Steps : {}  >  Loss : {}".format(
                    timedelta(seconds=(datetime.now() - start_time).total_seconds()), steps, running_loss / 2000))

        # Evaluation
        correct = 0
        total = 0
        with torch.no_grad():
            for steps, data, in enumerate(test_ds_loader):
                images, labels = data[0].to(gpu), data[1].to(gpu)  # use gpu
                pred = model(images)

                _, predicted = torch.max(pred.data, 1)  # FIX torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
       
        print("Accuracy : {:.0%}".format(correct / total))
 

if __name__ == '__main__':
    training('SGD', True)

 
