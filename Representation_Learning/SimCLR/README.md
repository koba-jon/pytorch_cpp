# SimCLR
This is the implementation of "SimCLR".<br>
Original paper: T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A Simple Framework for Contrastive Learning of Visual Representations. In Proceedings of the 37th International Conference on Machine Learning, 2020. [link](https://proceedings.mlr.press/v119/chen20j.html)

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
- Large-scale CelebFaces Attributes (CelebA) Dataset<br>
This is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.<br>
Link: [official](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

#### Setting

Please create a link for the dataset.<br>
The following hierarchical relationships are recommended.

~~~
datasets
|--Dataset1
|    |--train
|    |    |--image1.png
|    |    |--image2.bmp
|    |    |--image3.jpg
|    |
|    |--valid
|    |--test
|
|--Dataset2
|--Dataset3
~~~

You should substitute the path of training data for "<training_path>", test data for "<test_path>", respectively.<br>
The following is an example for "celebA".
~~~
$ cd datasets
$ mkdir celebA
$ cd celebA
$ ln -s <training_path> ./train
$ ln -s <test_path> ./test
$ cd ../..
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

DATA='celebA'

./SimCLR \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 224 \
    --batch_size 16 \
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

DATA='celebA'

./SimCLR \
    --test true \
    --dataset ${DATA} \
    --test_dir "test" \
    --size 224 \
    --gpu_id 0
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~

