# DAE2d
This is the implementation of "Denoising Autoencoder" corresponding to 2-dimensional shape.<br>
Original paper: P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th International Conference on Machine Learning, 2008. [link](https://dl.acm.org/doi/abs/10.1145/1390156.1390294)

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

./DAE2d \
    --train true \
    --RVIN true \
    --SPN false \
    --GN false \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --batch_size 16 \
    --gpu_id 0 \
    --nc 3
~~~
You can select the noise to be added to the input image by switching the "RVIN", "SPN", and "GN".

- Random Valued Impulse Noise (RVIN)<br>
This noise is generated from random number that follow uniform distribution in the range between the minimum and maximum pixel values.<br>
In addition, the final pixel value is replaced with the generated pixel value, which the original pixel value does not contribute the final one.

- Salt and Pepper Noise (SPN)<br>
This noise is generated from random number that follow Bernoulli distribution with the minimum and maximum pixel values.<br>
In addition, the final pixel value is replaced with the generated pixel value, which the original pixel value does not contribute the final one.

- Gaussian Noise (GN)<br>
This noise is generated from random number that follow normal distribution with a fixed mean and a fixed standard deviation.<br>
In addition, the final pixel value is the sum of the generated pixel value and the original pixel value.

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

./DAE2d \
    --test true \
    --RVIN true \
    --SPN false \
    --GN false \
    --dataset ${DATA} \
    --test_in_dir "test" \
    --test_out_dir "test" \
    --size 256 \
    --gpu_id 0 \
    --nc 3
~~~
If you want to test the denoising of real image, set "test_in_dir" to "directory of noisy images" and "test_out_dir" to "directory of output ground truth".<br>
In addition, you should turn off the flag of "RVIN", "SPN", and "GN".<br>
However, the two file names must correspond.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~

