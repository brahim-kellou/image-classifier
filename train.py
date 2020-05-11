import argparse
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import json

from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image

from utils import load_data


def build_model():
    print("Building model...")
    if (args.arch is None):
        arch = 'vgg'
    else:
        arch = args.arch

    if (arch == 'vgg'):
        model = models.vgg19(pretrained=True)
        input_node = 25088
    elif (arch == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_node = 1024

    if (args.hidden_units is None):
        hidden_units = 4096
    else:
        hidden_units = int(args.hidden_units)

    for param in model.parameters():
        param.requires_grad = False

    output_node = 102

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_node, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('fc2', nn.Linear(
                                                hidden_units, output_node)),
                                            ('relu', nn.ReLU()),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))

    model.classifier = classifier

    return model


def check_validation_set(model, loader, device='cpu'):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train(model, loaders):
    print("Training model...")

    if (args.learning_rate is None):
        learning_rate = 0.001
    else:
        learning_rate = float(args.learning_rate)

    if (args.epochs is None):
        epochs = 3
    else:
        epochs = int(args.epochs)

    if (args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'

    trainloader = loaders['train']
    validloader = loaders['valid']
    testloader = loaders['test']

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    print_every = 40
    steps = 0
    model.to(device)

    for e in range(epochs):
        print("-" * 10)
        print("Epoch: {}/{}...".format(e+1, epochs))
        running_loss = 0
        for _, (images, labels) in enumerate(trainloader):
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_accuracy = check_validation_set(
                    model, validloader, device)
                print("Epoch: {}".format(e+1),
                      ", Loss: {:.4f}".format(running_loss/print_every),
                      ", Validation Accuracy: {:.4f}".format(valid_accuracy))
                running_loss = 0

    print("Training Done!")

    return model


def save_model(model):
    print("Saving model...")

    if args.save_dir is None:
        save_dir = 'check.pth'
    else:
        save_dir = args.save_dir

    if args.arch is None:
        arch = 'vgg'
    else:
        arch = args.arch

    if arch == 'vgg':
        input_node = 25088
    else:
        input_node = 1024

    if args.hidden_units is None:
        hidden_units = 4096
    else:
        hidden_units = int(args.hidden_units)

    output_node = 102

    checkpoint = {
        'model': model.cpu(),
        'arch': arch,
        'input_node': input_node,
        'hidden_units': hidden_units,
        'output_node': output_node,
        'classifier': model.classifier,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, save_dir)

    return 0


def validate_inputs():
    print("Validating parameters...")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")

    if(not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')

    data_dir = os.listdir(args.data_directory)
    if (not set(data_dir).issubset({'test', 'train', 'valid'})):
        raise Exception('missing: test, train or valid sub-directories')

    if args.arch not in ('vgg', 'densenet', None):
        raise Exception('Please choose one of: vgg or densenet')


def parse():
    parser = argparse.ArgumentParser(
        description='Train a neural network for image classification')
    parser.add_argument('data_directory', help='data directory (required)')
    parser.add_argument(
        '--save_dir', help='directory to save a neural network.')
    parser.add_argument(
        '--arch', help='models to use OPTIONS[vgg, densenet]', default='vgg')
    parser.add_argument('--learning_rate',
                        help='learning rate', type=float, default=0.001)
    parser.add_argument(
        '--hidden_units', help='number of hidden units', type=int, default=4096)
    parser.add_argument('--epochs', help='epochs', type=int, default=3)
    parser.add_argument('--gpu', action='store_true',
                        help='gpu', default=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    print("Creating a deep learning model...")

    args = parse()
    validate_inputs()
    loaders = load_data(args.data_directory)
    model = build_model()
    model = train(model, loaders)
    save_model(model)

    print("Model finished!")
