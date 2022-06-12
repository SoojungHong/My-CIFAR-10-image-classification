import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

class DataLoader():
    
    def __init__(self, max_epochs=10, batch_size=4, transform=None):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def data_loader_train(self):
        train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return train_ds_loader

    def get_train_count(self):
        return len(train_ds)

    def data_loader_test(self):
        test_ds_loader = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return test_ds_loader

    def get_test_count(self):
        return len(test_ds)