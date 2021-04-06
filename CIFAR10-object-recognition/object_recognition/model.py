import torch
import torch.nn as nn
import torch.nn.functional as F

PATH = './weights/cifar10.pth'

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(1024, 516),
            nn.ReLU(),
            nn.Linear(516, 256),
            nn.ReLU(),
            nn.Linear(256, 10)            
        )

    def forward(self, x):
        return self.network(x)

    @staticmethod
    def add_to_argparse(parser, train=True):
        if train:
            parser.add_argument("--save_path", type=str, default=PATH)
        else:
            parser.add_argument("--load_path", type=str, required=True)
        
        return parser

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

model = Model().to("cuda")
