# CycleGAN
This is the implementation of "CycleGAN".<br>
Original paper: J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired Image-To-Image Translation Using Cycle-Consistent Adversarial Networks. In Proceedings of the IEEE International Conference on Computer Vision, 2017. [link](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)

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
- CMP Facade Database<br>
This is a dataset of facade images assembled at the Center for Machine Perception, which includes 606 rectified images of facades from various sources, which have been manually annotated.<br>
Link: [official](http://cmp.felk.cvut.cz/~tylecr1/facade/)

#### Setting

Please create a link for the dataset.<br>
The following hierarchical relationships are recommended.

~~~
datasets
|--Dataset1
|    |--trainA
|    |    |--image1.png
|    |    |--image2.bmp
|    |    |--image3.jpg
|    |
|    |--trainB
|    |    |--image4.png
|    |    |--image5.bmp
|    |    |--image6.jpg
|    |
|    |--validA
|    |--validB
|    |--testA
|    |--testB
|
|--Dataset2
|--Dataset3
~~~

You should substitute the path of training A data for "<training_A_path>", training B data for "<training_B_path>", test A data for "<test_A_path>", test B data for "<test_B_path>", respectively.<br>
The following is an example for "facade".
~~~
$ cd datasets
$ mkdir facade
$ cd facade
$ ln -s <training_A_path> ./trainA
$ ln -s <training_B_path> ./trainB
$ ln -s <test_A_path> ./testA
$ ln -s <test_B_path> ./testB
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

DATA='facade'

./CycleGAN \
    --train true \
    --epochs 300 \
    --iters 1000 \
    --dataset ${DATA} \
    --size 256 \
    --loss "vanilla" \
    --batch_size 1 \
    --gpu_id 0 \
    --A_nc 3 \
    --B_nc 3
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

DATA='facade'

./CycleGAN \
    --test true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --A_nc 3 \
    --B_nc 3
~~~
There are no particular restrictions on both input and output images.<br>
However, the two file names must correspond without the extension.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~


## Acknowledgments
This code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
