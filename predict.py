import argparse
import time
import torch
import numpy as np
import json
import sys

from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image

from utils import process_image


def load_model(checkpoint):
    model_info = torch.load(checkpoint)

    model = model_info['model']
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])

    return model


def classify_image(image_path, checkpoint, topk=5):
    topk = int(topk)
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model = load_model(checkpoint)
        if (args.gpu):
            image = image.cuda()
            model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)

        return probs[0].tolist(), classes[0].add(1).tolist()


def read_categories(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def display_prediction(results, category_names):
    cat_file = read_categories(category_names)
    for p, c in zip(results[0], results[1]):
        c = cat_file.get(str(c), 'None')
        print("{} , {:.2f}".format(c, p))
    return None


def validate_inputs():
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")


def parse():
    parser = argparse.ArgumentParser(
        description='Neural network to classify an image')
    parser.add_argument(
        'input', help='Image file to classifiy (required)')
    parser.add_argument(
        'checkpoint', help='Model used for classification (required)')
    parser.add_argument(
        '--top_k', help='How many prediction categories to show.', default=5)
    parser.add_argument(
        '--category_names', help='File for category names', default="cat_to_name.json")
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    args = parser.parse_args()
    return args


def main():
    global args
    args = parse()
    validate_inputs()

    input = args.input
    checkpoint = args.checkpoint
    top_k = args.top_k

    prediction = classify_image(input, checkpoint, top_k)
    display_prediction(prediction, args.category_names)

    return prediction


main()
