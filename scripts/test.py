"""
This file tests the pretrained PyTorch image classification model (.pth file)
"""


import torch
from model import ImageClassifier

import glob
import cv2
import numpy as np


def test():
    # Create network
    net = ImageClassifier()
    PATH = './models/image_classifier.pth'

    # Load the PyTorch model
    net.load_state_dict(torch.load(PATH))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net.to(device)

    # Class names
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Inference on images
    net.eval()
    with torch.no_grad():
        for f in glob.glob('./images/*'):
            # Load image
            img = cv2.imread(f, 1)

            # Preprocessing
            img.astype(float)
            img = torch.unsqueeze(torch.from_numpy(np.transpose(img, (2, 0, 1))), dim=0)    # input shape: nchw
            img = 2*(img / 255.0 - 0.5)

            # Inference
            img = img.to(device)
            output = net(img)      # output shape: n*n_class

            # Get prediction
            _, prediction = torch.max(output, 1)
            print(f"The input image {f}")
            print(f"The predicted class is {classes[prediction.item()]}")


if __name__ == '__main__':
    test()