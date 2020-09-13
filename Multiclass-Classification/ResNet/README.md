# ResNet
This is the implementation of "ResNet" for Multiclass-Classification.<br>
Original paper: K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceesings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016. [link](http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

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
$ sh set_MNIST.sh
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

./ResNet \
    --train true \
    --n_layers 50 \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 224 \
    --batch_size 16 \
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

./ResNet \
    --test true \
    --n_layers 50 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 10 \
    --size 224 \
    --gpu_id 0 \
    --nc 1
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~


## Acknowledgments
This code is inspired by [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

