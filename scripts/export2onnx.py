
"""
This file converts the pretrained PyTorch model (.pth file) to the onnx model (.onnx file)
"""


import torch
import torch.backends.cudnn as cudnn

from model import ImageClassifier


cudnn.benchmark = True


def convert():
    # Create the model
    net = ImageClassifier()
    net.cuda()

    # Load model parameters
    PATH = './models/image_classifier.pth'
    net.load_state_dict(torch.load(PATH))
    net.eval()

    # Create input
    x = torch.randn((1, 3, 32, 32)).cuda()    # input format: nchw

    # Export to onnx model
    print("Converting to onnx...")
    torch.onnx.export(net,                             # model being run
                x,                                     # model input
                "./models/image_classifier.onnx",      # where to save the model
                export_params=True,                    # store the trained parameter weights inside the model file
                opset_version=11,                      # the ONNX version to export the model to
                do_constant_folding=True,              # use constant folding optimization
                input_names= ['input'],                # specify the model input names 
                output_names=['output'],               # specify the model output names
                )
    print("Done! Onnx output model: image_classifier.onnx")


if __name__ == '__main__':
    convert()
