# PNDM2d-v
This is the implementation of "PNDM" corresponding to 2-dimensional shape based on v-prediction for high resolution.<br>
Original paper: L. Liu, Y. Ren, Z. Lin, and Z. Zhao. Pseudo Numerical Methods for Diffusion Models on Manifolds. In International Conference on Learning Representations, 2022. [link](https://arxiv.org/abs/2202.09778)

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

./PNDM2d-v \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --size 256 \
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

./PNDM2d-v \
    --test true \
    --dataset ${DATA} \
    --test_in_dir "test" \
    --test_out_dir "test" \
    --size 256 \
    --gpu_id 0 \
    --nc 3
~~~
If you want to test the reconstruction error of the image, the above settings will work.<br>
If you want to test the denoising of the image, set "test_in_dir" to "directory of noisy images" and "test_out_dir" to "directory of output ground truth".<br>
However, the two file names must correspond.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~

### 5. Image Synthesis

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/synth.sh
~~~
The following is an example of the synthesis phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='celebA'

./PNDM2d-v \
    --synth true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
~~~
If you want to generate image, the above settings will work.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/synth.sh
~~~

### 6. Image Sampling

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/sample.sh
~~~
The following is an example of the sampling phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='celebA'

./PNDM2d-v \
    --sample true \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 256 \
    --gpu_id 0 \
    --nc 3
~~~
If you want to generate image, the above settings will work.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/sample.sh
~~~

