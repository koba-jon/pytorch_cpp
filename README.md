# PyTorch C++ Samples
These are Deep Learning sample programs of PyTorch written in C++.

## Description
PyTorch is famous as a kind of Deep Learning Frameworks.<br>
Among them, Python source code is overflowing on the Web, so we can easily write the source code of Deep Learning in Python.<br>
However, there is very little source code written in C++ of compiler language.<br>
Therefore, I hope this repository will help many programmers by providing PyTorch sample programs written in C++.<br>
In addition, I might adapt programs to the latest version. <br>

## Implementation

### Multiclass-Classification

<table>
  <tr>
    <th>Model</th>
    <th>Paper</th>
    <th>Conference/Journal</th>
    <th>Code</th>
    <th>Release Version</th>
  </tr>
  <tr>
    <td>AlexNet</td>
    <td><a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ">A. Krizhevsky et al.</a></td>
    <td>NeurIPS 2012</td>
    <td><a href="Multiclass-Classification/AlexNet">AlexNet</a></td>
    <td><b>v1.6.1(Latest)</b></td>
  </tr>
  <tr>
    <td>VGGNet</td>
    <td><a href="https://arxiv.org/abs/1409.1556">K. Simonyan et al.</a></td>
    <td>ICLR 2015</td>
    <td><a href="Multiclass-Classification/VGGNet">VGGNet</a></td>
    <td><b>v1.6.1(Latest)</b></td>
  </tr>
</table>

### Autoencoders

<table>
  <tr>
    <th>Model</th>
    <th>Paper</th>
    <th>Conference/Journal</th>
    <th>Code</th>
    <th>Release Version</th>
  </tr>
  <tr>
    <td>Autoencoder</td>
    <td><a href="https://science.sciencemag.org/content/313/5786/504.abstract">G. E. Hinton et al.</a></td>
    <td>Science 2006</td>
    <td><a href="AE2d">AE2d</a></td>
    <td>v1.5.0</td>
  </tr>
  <tr>
    <td>Variational Autoencoder</td>
    <td><a href="https://arxiv.org/abs/1312.6114">D. P. Kingma et al.</a></td>
    <td>ICLR 2014</td>
    <td><a href="VAE2d">VAE2d</a></td>
    <td>v1.5.1</td>
  </tr>
  <tr>
    <td rowspan="2">Wasserstein Autoencoder</td>
    <td rowspan="2"><a href="https://openreview.net/forum?id=HkL7n1-0b">I. Tolstikhin et al.</a></td>
    <td rowspan="2">ICLR 2018</td>
    <td><a href="WAE2d/GAN">WAE2d GAN</a></td>
    <td rowspan="2"><b>v1.6.1(Latest)</b></td>
  </tr>
  <tr>
    <td><a href="WAE2d/MMD">WAE2d MMD</a></td>
  </tr>
</table>

### Encoder-Decoder Networks

<table>
  <tr>
    <th>Model</th>
    <th>Paper</th>
    <th>Conference/Journal</th>
    <th>Code</th>
    <th>Release Version</th>
  </tr>
  <tr>
    <td rowspan="2">U-Net</td>
    <td rowspan="2"><a href="https://arxiv.org/abs/1505.04597">O. Ronneberger et al.</a></td>
    <td rowspan="2">MICCAI 2015</td>
    <td><a href="U-Net/Classification">U-Net Classification</a></td>
    <td rowspan="2">v1.5.1</td>
  </tr>
  <tr>
    <td><a href="U-Net/Regression">U-Net Regression</a></td>
  </tr>
</table>

### Generative Adversarial Networks

<table>
  <tr>
    <th>Model</th>
    <th>Paper</th>
    <th>Conference/Journal</th>
    <th>Code</th>
    <th>Release Version</th>
  </tr>
  <tr>
    <td>DCGAN</td>
    <td><a href="https://arxiv.org/abs/1511.06434">A. Radford et al.</a></td>
    <td>ICLR 2016</td>
    <td><a href="DCGAN">DCGAN</a></td>
    <td>v1.5.1</td>
  </tr>
  <tr>
    <td>pix2pix</td>
    <td><a href="https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html">P. Isola et al.</a></td>
    <td>CVPR 2017</td>
    <td><a href="pix2pix">pix2pix</a></td>
    <td>v1.5.1</td>
  </tr>
</table>

### Anomaly Detection

<table>
  <tr>
    <th>Model</th>
    <th>Paper</th>
    <th>Conference/Journal</th>
    <th>Code</th>
    <th>Release Version</th>
  </tr>
  <tr>
    <td>DAGMM</td>
    <td><a href="https://openreview.net/forum?id=BJJLHbb0-">B. Zong et al.</a></td>
    <td>ICLR 2018</td>
    <td><a href="DAGMM2d">DAGMM2d</a></td>
    <td>v1.6.0</td>
  </tr>
</table>

## Requirement

### 1. PyTorch C++
Please select the environment to use as follows on PyTorch official. <br>
PyTorch official : https://pytorch.org/ <br>
***
PyTorch Build : Preview (Nightly) <br>
Your OS : Linux <br>
Package : LibTorch <br>
CUDA : 10.2 <br>
Run this Command : Download here (cxx11 ABI) <br>
GPU : https://download.pytorch.org/libtorch/nightly/cu102/libtorch-cxx11-abi-shared-with-deps-latest.zip <br>
CPU : https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip <br>
***

### 2. OpenCV
version : 3.0.0 or more <br>
This is used for pre-processing and post-processing. <br>
Please refer to other sites for more detailed installation method.

### 3. OpenMP
This is used to load data in parallel. <br>
(It may be installed on standard Linux OS.)

### 4. Boost
This is used for command line arguments, etc. <br>
~~~
$ sudo apt install libboost-dev libboost-all-dev
~~~

### 5. Gnuplot
This is used to display loss graph. <br>
~~~
$ sudo apt install gnuplot
~~~

### 6. libpng/png++
This is used to load and save index-color image in semantic segmentation. <br>
~~~
$ sudo apt install libpng-dev libpng++-dev
~~~

## Preparation

### 1. Git Clone
~~~
$ git clone -b develop/v1.6.1 https://github.com/koba-jon/pytorch_cpp.git
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
$ ln -s <dataset_path> ./celebA_org
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
$ cd ../..
~~~

### 2. Data Input System
There are transform, dataset and dataloader for data input in this repository.<br>
It corresponds to the following source code in the directory, and we can add new function to the source code below.
- transforms.cpp
- transforms.hpp
- datasets.cpp
- datasets.hpp
- dataloader.cpp
- dataloader.hpp

### 3. Check Progress
There are a feature to check progress for training in this repository.<br>
We can watch the number of epoch, loss, time and speed in training.<br>
![util1](https://user-images.githubusercontent.com/56967584/88464264-3f720300-cef4-11ea-85fd-360cb3a424d1.png)<br>
It corresponds to the following source code in the directory.
- progress.cpp
- progress.hpp

### 4. Monitoring System
There are monitoring system for training in this repository.<br>
We can watch output image and loss graph.<br>
The feature to watch output image is in the "samples" in the directory "checkpoints" created during training.<br>
The feature to watch loss graph is in the "graph" in the directory "checkpoints" created during training.<br>
![util2](https://user-images.githubusercontent.com/56967584/88464268-40a33000-cef4-11ea-8a3c-da42d4c803b6.png)<br>
It corresponds to the following source code in the directory.
- visualizer.cpp
- visualizer.hpp

## Conclusion
I hope this repository will help many programmers by providing PyTorch sample programs written in C++.<br>
If you have any problems with the source code of this repository, please feel free to "issue".<br>
Let's have a good development and research life!


## License
You can feel free to use all source code in this repository.<br>
(Click [here](LICENSE) for details.)<br><br>

But if you exploit external libraries (e.g. redistribution), you should be careful.<br>
At a minimum, the license notation at the following URL is required.<br>
In addition, third party copyrights belong to their respective owners.<br>

- PyTorch <br>
Official : https://pytorch.org/ <br>
License : https://github.com/pytorch/pytorch/blob/master/LICENSE <br>

- OpenCV <br>
Official : https://opencv.org/ <br>
License : https://opencv.org/license/ <br>

- OpenMP <br>
Official : https://www.openmp.org/ <br>
License : https://gcc.gnu.org/onlinedocs/ <br>

- Boost <br>
Official : https://www.boost.org/ <br>
License : https://www.boost.org/users/license.html <br>

- Gnuplot <br>
Official : http://www.gnuplot.info/ <br>
License : https://sourceforge.net/p/gnuplot/gnuplot-main/ci/master/tree/Copyright <br>

- libpng/png++ <br>
Official (libpng) : http://www.libpng.org/pub/png/libpng.html <br>
License (libpng) : http://www.libpng.org/pub/png/src/libpng-LICENSE.txt <br>
Official (png++) : https://www.nongnu.org/pngpp/ <br>
License (png++) : https://www.nongnu.org/pngpp/license.html <br>
