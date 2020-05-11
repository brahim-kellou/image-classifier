import numpy as np
import torch
import json

from torchvision import datasets, transforms
from PIL import Image


def load_data(data_dir):
    print("Loading data...")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomRotation(180),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                           (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                           ])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                          (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                          ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train': trainloader, 'valid': validloader,
               'test': testloader, 'labels': cat_to_name}

    return loaders


def process_image(image):
    image = Image.open(image)
    width, height = image.size
    picture_coords = [width, height]
    max_span = max(picture_coords)
    max_element = picture_coords.index(max_span)
    if (max_element == 0):
        min_element = 1
    else:
        min_element = 0
    aspect_ratio = picture_coords[max_element]/picture_coords[min_element]
    new_picture_coords = [0, 0]
    new_picture_coords[min_element] = 256
    new_picture_coords[max_element] = int(256 * aspect_ratio)
    image = image.resize(new_picture_coords)
    width, height = new_picture_coords
    left = (width - 244)/2
    top = (height - 244)/2
    right = (width + 244)/2
    bottom = (height + 244)/2
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image)
    np_image = np_image.astype('float64')
    np_image = np_image / [255, 255, 255]
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))

    return np_image
