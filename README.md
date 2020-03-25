# PyTorch C++ Samples
These are Deep Learning sample programs of PyTorch written in C++.

## Description
PyTorch is famous as a kind of Deep Learning Frameworks.<br>
Among them, Python source code is overflowing on the Web, so we can easily write the source code of Deep Learning in Python.<br>
However, there is very little source code written in C++ of compiler language.<br>
Therefore, I hope this repository will help many programmers by providing PyTorch sample programs written in C++.<br>
In addition, I might adapt programs to the latest version. <br>

## Requirement

### 1. PyTorch C++
Please select the environment to use as follows on PyTorch official. <br>
PyTorch official : https://pytorch.org/ <br>
***
PyTorch Build : Preview (Nightly) <br>
Your OS : Linux <br>
Package : LibTorch <br>
CUDA : 10.1 <br>
Run this Command : https://download.pytorch.org/libtorch/nightly/cu101/libtorch-cxx11-abi-shared-with-deps-latest.zip <br>
***

### 2. OpenCV
version : 3.0.0 or more <br>
This is used for pre-processing and post-processing. <br>
Please refer to other sites for more detailed installation method.

### 3. OpenMP
version : 4.5 (There is a possibility that other versions can do it.) <br>
This is used to load data in parallel. <br>
It may be installed on standard Linux OS.

### 4. Boost
version : 1.65.1 (There is a possibility that other versions can do it.) <br>
This is used for command line arguments, etc. <br>
~~~
$ sudo apt install libboost-dev libboost-all-dev
~~~

### 5. Gnuplot
version : 5.2 (There is a possibility that other versions can do it.) <br>
This is used to display loss graph. <br>
~~~
$ sudo apt install gnuplot
~~~

## Usage

### 1. Git Clone
~~~
$ git clone https://github.com/koba-jon/pytorch_cpp.git
$ cd pytorch_cpp
~~~

### 2. Path Setting
The following is an example of "ConvAE".
~~~
$ cd ConvAE
$ vi CMakeLists.txt
~~~
Please change the 7th line of "CMakeLists.txt" according to the path of the directory "libtorch". <br>
The following is an example where the directory "libtorch" is located directly under the directory "HOME".
~~~
6: # LibTorch
7: set(LIBTORCH_DIR $ENV{HOME}/libtorch)
8: list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
~~~

### 3. Build
Please build the source file according to the procedure.
~~~
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ cd ..
~~~

### 4. Dataset Setting
Please create a link for the dataset.
~~~
$ cd datasets
$ ln -s <dataset_path> ./
~~~
Please edit the file for original dataset.
~~~
$ vi hold_out.sh
~~~
The following is an example for "celebA".
~~~
#!/bin/bash

python3 hold_out.py \
        --input_dir celebA_org \
        --output_dir celebA \
        --train_rate 9 \
        --valid_rate 1
~~~
By running this file, you can split it into training and validation data.
~~~
$ sudo apt install python3 python3-pip
$ pip3 install natsort
$ sh hold_out.sh
$ cd ..
~~~

### 5. Execute File Setting
Please set the shell for executable file.
~~~
$ cd scripts
$ vi train.sh
~~~
The following is an exmaple of the learning phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='celebA'

./ConvAE \
        --train true \
        --epochs 300 \
        --dataset ${DATA} \
        --size 256 \
        --loss "l1" \
        --batch_size 16 \
        --gpu_id 0 \
        --nc 3
~~~
~~~
$ cd ..
~~~

### 6. Run
Please execute the following to start the Deep Learning program.
~~~
$ sh scripts/train.sh
~~~
