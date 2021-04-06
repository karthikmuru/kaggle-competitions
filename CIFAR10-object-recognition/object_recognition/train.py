import torch
import torch.optim as optim
import torch.nn as nn

import argparse
import numpy as np
from data import Data
from model import model
from data import Utils

def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    
    model_group = parser.add_argument_group("Model Args")
    model.add_to_argparse(model_group)

    return parser

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data = Data(args.batch_size)

    print("Model Summary : ")
    print(model)
    
    train(args, data)
    model.save(args.save_path)

def train(args, data):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.epochs):
        running_loss = 0.0

        for i, d in enumerate(data.trainloader, 0):

            inputs, labels = d
            inputs, labels = inputs.cuda(), labels.cuda()
            # Setting the gradients to 0
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            # Calculate the gradient based on all the parameters
            loss.backward()
            # Update all the parameters based on the gradients
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

if __name__ == "__main__":
    main()
