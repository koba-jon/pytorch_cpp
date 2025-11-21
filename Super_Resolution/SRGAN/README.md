# SRGAN
This is the implementation of "Super Resolution Generative Adversarial Network".<br>
Original paper: C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, and W. Shi. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. [link](https://openaccess.thecvf.com/content_cvpr_2017/html/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.html)

## Usage

### 0. Download pre-trained model
Please download VGG19 pre-trained model with ImageNet.
~~~
$ wget https://github.com/koba-jon/pytorch_cpp/releases/download/vgg19/vgg19.pth
~~~

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
|    |--trainLR
|    |    |--image1.png
|    |    |--image2.bmp
|    |    |--image3.jpg
|    |
|    |--trainHR
|    |    |--image1.png
|    |    |--image2.bmp
|    |    |--image3.jpg
|    |
|    |--validLR
|    |--validHR
|    |--testLR
|    |--testHR
|
|--Dataset2
|--Dataset3
~~~

You should substitute the path of training low-resolution image for "<training_lr_path>", training high-resolution image for "<training_hr_path>", test low-resolution image for "<test_lr_path>", test high-resolution image for "<test_hr_path>", respectively.<br>
The following is an example for "celebA".
~~~
$ cd datasets
$ mkdir celebA
$ cd celebA
$ ln -s <training_lr_path> ./trainLR
$ ln -s <training_hr_path> ./trainHR
$ ln -s <test_lr_path> ./testLR
$ ln -s <test_hr_path> ./testHR
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

./SRGAN \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --hr_size 128 \
    --loss "vanilla" \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
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

./SRGAN \
    --test true \
    --dataset ${DATA} \
    --hr_size 128 \
    --gpu_id 0 \
    --nc 3
~~~
There are no particular restrictions on both input and output images.<br>
However, the two file names must correspond without the extension.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~


