import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math

from datetime import datetime, timedelta
from data_loader import DataLoader
from model import *
from optimizer import Optimizer
from loss import Loss

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if not torch.cuda.is_available():
    raise Exception('At least one gpu must be available')
gpu = torch.device('cuda')


def training(data_loader, enable_BN = False, opti='ADAM', optimizer=None, loss=None):

    # select model
    if enable_BN == True:
        model = BN_Model()
    else:
        model = Model()

    model.to(gpu)

    # select optimizer
    if opti == 'SGD':
        optimizer = optimizer.sgd_optimizer_fn(model)
    elif opti == 'ADAM':
        optimizer = optimizer.adam_optimizer_fn(model)
    
    loss_fn = loss.criterion() 
    start_time = datetime.now()
    train_count = data_loader.get_train_count() 
    for e in range(max_epochs):
        print('epoch : ', e)
        steps = e * train_count 
        running_loss = 0.0
        for steps, data, in enumerate(data_loader.data_loader_train(), 0): 
            images, labels = data[0].to(gpu), data[1].to(gpu)  
            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % 1000 == 0: 
                print("Esapsed Time : {}s  >  Steps : {}  >  Loss : {}".format(
                    timedelta(seconds=(datetime.now() - start_time).total_seconds()), steps, running_loss / 2000))
        
        # Evaluation
        correct = 0
        total = 0
        with torch.no_grad():
            for steps, data, in enumerate(data_loader. data_loader_test()):
                images, labels = data[0].to(gpu), data[1].to(gpu) 
                pred = model(images)

                _, predicted = torch.max(pred.data, 1)  
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print("Accuracy : {:.0%}".format(correct / total))
    

if __name__ == '__main__':
     max_epochs = 10
     batch_size = 4
     data_loader = DataLoader(max_epochs, batch_size)
     optimizer = Optimizer 
     loss = Loss 
     training(data_loader, True, 'SGD', optimizer, loss)

