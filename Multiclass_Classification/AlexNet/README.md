# AlexNet
This is the implementation of "AlexNet" for Multiclass Classification.<br>
Original paper: A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In Proceedings of the 25th International Conference on Neural Information Processing Systems, 2012. [link](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ)

## Usage

### 1. Build
Please build the source file according to the procedure.
~~~
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ cd ..
~~~

### 2. Dataset Setting

#### Recommendation
- THE MNIST DATABASE of handwritten digits<br>
This is the dataset of 28x28 grayscale for handwritten digits in 10 classes that has a training set of 60000 images and a test set of 10000 images.<br>
Link: [official](http://yann.lecun.com/exdb/mnist/)

- The CIFAR-10 dataset<br>
This is the dataset of 32x32 color based on labeled tiny images in 10 classes that has a training set of 50000 images and a test set of 10000 images.<br>
Link: [official](https://www.cs.toronto.edu/~kriz/cifar.html)

- The CIFAR-100 dataset<br>
This is the dataset of 32x32 color based on labeled tiny images in 100 classes that has a training set of 50000 images and a test set of 10000 images.<br>
Link: [official](https://www.cs.toronto.edu/~kriz/cifar.html)

#### Setting

Please create a link for the dataset.<br>
The following hierarchical relationships are recommended.

~~~
datasets
|--Dataset1
|    |--train
|    |    |--class1
|    |    |    |--image1.png
|    |    |    |--image2.bmp
|    |    |    |--image3.jpg
|    |    |
|    |    |--class2
|    |    |--class3
|    |
|    |--valid
|    |--test
|
|--Dataset2
|--Dataset3
~~~

The following is an example for "MNIST".<br>
This is downloaded and placed, maintaining the above hierarchical relationships.
~~~
$ cd datasets
$ sudo apt install python3 python3-pip
$ pip3 install scikit-image
$ sh ../../../scripts/set_MNIST.sh
$ cd ..
~~~

Please set the text file for class names.
~~~
$ vi list/MNIST.txt
~~~

In case of "MNIST", please set as follows.
~~~
0
1
2
3
4
5
6
7
8
9
~~~

### 3. Training

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/train.sh
~~~
The following is an example of the training phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='MNIST'

./AlexNet \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 227 \
    --batch_size 64 \
    --gpu_id 0 \
    --nc 1
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/train.sh
~~~

### 4. Test

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/test.sh
~~~
The following is an example of the test phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='MNIST'

./AlexNet \
    --test true \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 227 \
    --gpu_id 0 \
    --nc 1
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~


## Acknowledgments
This code is inspired by [alexnet-pytorch](https://github.com/dansuh17/alexnet-pytorch).

