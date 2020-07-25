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
CUDA : 10.2 <br>
Run this Command :
    Download here (cxx11 ABI): https://download.pytorch.org/libtorch/nightly/cu102/libtorch-cxx11-abi-shared-with-deps-latest.zip <br>
***

### 2. OpenCV
version : 3.0.0 or more <br>
This is used for pre-processing and post-processing. <br>
Please refer to other sites for more detailed installation method.

### 3. OpenMP
version : 4.5 (It works on other versions.) <br>
This is used to load data in parallel. <br>
It may be installed on standard Linux OS.

### 4. Boost
version : 1.65.1 (It works on other versions.) <br>
This is used for command line arguments, etc. <br>
~~~
$ sudo apt install libboost-dev libboost-all-dev
~~~

### 5. Gnuplot
version : 5.2 (It works on other versions.) <br>
This is used to display loss graph. <br>
~~~
$ sudo apt install gnuplot
~~~

### 6. libpng
version : 0.2 (It works on other versions.) <br>
This is used to load and save index-color image in semantic segmentation. <br>
~~~
$ sudo apt install libpng++-dev
~~~

## Preparation

### 1. Git Clone
~~~
$ git clone -b develop/v1.6.0 https://github.com/koba-jon/pytorch_cpp.git
$ cd pytorch_cpp
~~~

### 2. Path Setting
~~~
$ vi utils/CMakeLists.txt
~~~
Please change the 4th line of "CMakeLists.txt" according to the path of the directory "libtorch". <br>
The following is an example where the directory "libtorch" is located directly under the directory "HOME".
~~~
3: # LibTorch
4: set(LIBTORCH_DIR $ENV{HOME}/libtorch)
5: list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
~~~

### 3. Execution
Please move to the directory of each model and refer to "README.md".

## Utility

### 1. Making Original Dataset
Please create a link for the original dataset.<br>
The following is an example of "AE2d" using "celebA" Dataset.
~~~
$ cd AE2d/datasets
$ ln -s <dataset_path> ./
~~~
You should substitute the path of dataset for "<dataset_path>".<br>
Please make sure you have training or test data directly under "<dataset_path>".
~~~
$ vi hold_out.sh
~~~
Please edit the file for original dataset.
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

### 2. Data Input/Output
There are transform, dataset and dataloader for data input/output in this repository.<br>
We can add new function to the source code below.<br>
It corresponds to the following source code in the directory.
- transforms.cpp
- transforms.hpp
- datasets.cpp
- datasets.hpp
- dataloader.cpp
- dataloader.hpp

### 2. Check Progress
There are a feature to check progress for training progress in this repository.<br>
We can watch number of epoch, loss, time and speed in training.<br>
![util1](https://user-images.githubusercontent.com/56967584/88464264-3f720300-cef4-11ea-85fd-360cb3a424d1.png)
It corresponds to the following source code in the directory.
- progress.cpp
- progress.hpp

### 3. Monitoring System
There are monitoring system for training in this repository.<br>
We can watch output image and loss graph.<br>
The feature to watch output image is in the "samples" in the directory "checkpoints" created during training.<br>
The feature to watch loss graph is in the "graph" in the directory "checkpoints" created during training.<br>
![util2](https://user-images.githubusercontent.com/56967584/88464268-40a33000-cef4-11ea-8a3c-da42d4c803b6.png)
It corresponds to the following source code in the directory.
- visualizer.cpp
- visualizer.hpp

## Conclusion
I hope this repository will help many programmers by providing PyTorch sample programs written in C++.<br>
If you have any problems with the source code of this repository, please feel free to "issue".<br>
Let's have a good development and research life!
