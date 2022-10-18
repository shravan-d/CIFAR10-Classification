# CIFAR10-Classification

Classification model for the CIFAR10 dataset. 

## Description
The CIFAR10 dataset contains 60000 32x32 dimensional images consisting of 10 different classes. It is widely considered a "solved" problem with powerful networks like the ResNet obtaining over 97% accuracy. What i have provided here is a less powerful model in terms of depth and computation, coming at the cost of accuracy. The model follows the CNN architecture with 10 convolutional layers and 3 fully connected layers. Each convolutional layer has a 3x3 filter, and is followed by Relu activation and batch nomralisation. After each convolutional layer, max pooling is used. The dense network output 10 scores, one for each of the classes we have. The learning rate, batch size and epoch number were fine tuned to provide the best output.
![Trainig result](ResultScreencap)

### Requirements
* Python 3.x
* Pytorch
* Torchvision
* OpenCV
* Numpy

### Installing
```
git clone https://github.com/shravan-d/CIFAR10-Classification.git
```

### Executing program

Training
```
python classify.py train
```
Predicting
```
python classify.py test frog_sample_4.png
```

## Acknowledgments
* [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
* [Dataset Page](https://www.cs.toronto.edu/~kriz/cifar.html)
