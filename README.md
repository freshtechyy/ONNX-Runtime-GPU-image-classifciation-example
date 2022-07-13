# Image Classification Using ONNX Runtime

Image classification example using ONNX Runtime C++ with CUDA.

## Dependencies

* ONNX Runtime
* CMake 3.23.1
* OpenCV 4.5.2
* CUDA 11.4
* CUDNN
* Python 3.8.10
* PyTorch 1.9.1

## ONNX Runtime GPU Installation

```bash
$ git clone --recursive https://github.com/Microsoft/onnxruntime
$ cd onnxruntime/
$ ./build.sh --skip_tests --use_cuda --config Release --build_shared_lib --parallel --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda
```

Change *ONNXRUNTIME_ROOT_PATH* in CMakeLists.txt to your ONNX Runtime installation directory.

## Usages

### Training

```bash
$ git clone 
$ cd onnxruntime_image_classification/
$ python3 scripts/train.py
```

After training, the pretrained PyTorch model (image_classifier.pth) is stored in *models* directory.

### Test on the pretrained PyTorch model

```bash
$ python3 scripts/text.py
```

The following results will be shown in the terminal.

```bash
The input image ./images/bird2.png
The predicted class is bird
The input image ./images/car.png
The predicted class is car
The input image ./images/bird1.png
The predicted class is bird
The input image ./images/horse.png
The predicted class is horse
```

### Convert the pretrained PyTorch model to ONNX model

```bash
$ python3 scripts/export2onnx.py
```

After the conversion, the ONNX model (image_classifier.onnx) is stored in *models* directory.

### Image classification inference in C++

```bash
$ mkdir build && cd build
$ cmake ..
$ make
$ ./src/image_classifier
```

The following results will be shown in the terminal.

```bash
******* Model information below *******
Number of Input Nodes: 1
Input Name: input
Input Type: float
Input Dimensions: [1, 3, 32, 32]
Number of Output Nodes: 1
Output Name: output
Output Type: float
Output Dimensions: [1, 10]

******* Predicition results below *******
Loaded image: ../images/bird2.png
Predicted class: bird

Loaded image: ../images/car.png
Predicted class: car

Loaded image: ../images/bird1.png
Predicted class: bird

Loaded image: ../images/horse.png
Predicted class: horse
```

## References

* https://onnxruntime.ai/
* https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
