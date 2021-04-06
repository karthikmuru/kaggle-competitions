import torch
import torchvision
import torchvision.transforms as transforms

class Data():

    def __init__(self, batch_size = 4, num_workers = 2):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self._transform())
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self._transform())

        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def _transform(self):
        return transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def classes(self):
        return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')