# AE1d
This is the implementation of "Autoencoder" corresponding to 1-dimensional shape.<br>
Original paper: G. E. Hinton and R. R. Salakhutdinov. Reducing the dimensionality of data with neural networks. Science, 2006. [link](https://science.sciencemag.org/content/313/5786/504.abstract)

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
- Normal Distribution Dataset<br>
This is the dataset of 300-dimensional data of 1-dimensional shape that has a training set of 100,000 pieces and a test set of 1,000 pieces.<br>
Link: [official](https://github.com/koba-jon/normal_distribution_dataset)

- THE MNIST DATABASE of handwritten digits<br>
This is the dataset of 28x28 grayscale for handwritten digits in 10 classes that has a training set of 60000 images and a test set of 10000 images.<br>
Link: [official](http://yann.lecun.com/exdb/mnist/)

#### Setting

Please create a link for the dataset.<br>
The following hierarchical relationships are recommended.

~~~
datasets
|--Dataset1
|    |--train
|    |    |--data1.dat
|    |    |--data2.csv
|    |    |--data3.txt
|    |
|    |--valid
|    |--test
|
|--Dataset2
|--Dataset3
~~~

The following is an example for "NormalDistribution".
~~~
$ cd datasets
$ git clone https://github.com/koba-jon/normal_distribution_dataset.git
$ ln -s normal_distribution_dataset/NormalDistribution ./NormalDistribution
$ cd ..
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

DATA='NormalDistribution'

./AE1d \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --nd 300 \
    --nz 1 \
    --batch_size 64 \
    --gpu_id 0
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

DATA='NormalDistribution'

./AE1d \
    --test true \
    --dataset ${DATA} \
    --test_in_dir "test" \
    --test_out_dir "test" \
    --nd 300 \
    --nz 1 \
    --gpu_id 0
~~~
If you want to test the reconstruction error of the data, the above settings will work.<br>
If you want to test the denoising of the data, set "test_in_dir" to "directory of noisy data" and "test_out_dir" to "directory of output ground truth".<br>
However, the two file names must correspond.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~

