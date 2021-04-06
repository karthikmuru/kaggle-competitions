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

    model_group = parser.add_argument_group("Model Args")
    model.add_to_argparse(model_group, train=False)

    return parser

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data = Data()
    model.load(args.load_path)

    evaluate(data)

def evaluate(data):

    correct = 0
    total = 0

    with torch.no_grad():
        for test_data in data.testloader: 
            images, labels = test_data

            outputs = model(images)

            _, predictions = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    print('Accuracy : %d %%' % (100 * correct / total))


if __name__ == "__main__":
    main()