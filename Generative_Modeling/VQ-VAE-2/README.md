# VQ-VAE-2
This is the implementation of "VQ-VAE-2".<br>
Original paper: A. Razavi, A. v. d. Oord, and O. Vinyals. Generating Diverse High-Fidelity Images with VQ-VAE-2. In Annual Conference on Neural Information Processing Systems, 2019. [link](https://proceedings.neurips.cc/paper/2019/hash/5f8e2fa1718d1bbcadf1cd9c7a54fb8c-Abstract.html)

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
|    |--train1
|    |    |--image1.png
|    |    |--image2.bmp
|    |    |--image3.jpg
|    |
|    |--valid1
|    |--test1
|    |--train2
|    |    |--image1.png
|    |    |--image2.bmp
|    |    |--image3.jpg
|    |
|    |--valid2
|    |--test2
|    |--train3
|    |    |--image1.png
|    |    |--image2.bmp
|    |    |--image3.jpg
|    |
|    |--valid3
|    |--test3
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
$ ln -s <training_path> ./train1
$ ln -s <training_path> ./train2
$ ln -s <training_path> ./train3
$ ln -s <test_path> ./test1
$ ln -s <test_path> ./test2
$ ln -s <test_path> ./test3
$ cd ../..
~~~

### 3.1. Training 1 (VQ-VAE-2)

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/train1.sh
~~~
The following is an example of the training phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='celebA'

./VQ-VAE-2 \
    --train1 true \
    --train1_epochs 300 \
    --dataset ${DATA} \
    --size 256 \
    --train1_batch_size 16 \
    --gpu_id 0 \
    --nc 3
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/train1.sh
~~~

### 3.2. Test 1 (VQ-VAE-2)

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/test1.sh
~~~
The following is an example of the test phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='celebA'

./VQ-VAE-2 \
    --test1 true \
    --dataset ${DATA} \
    --test1_in_dir "test1" \
    --test1_out_dir "test1" \
    --size 256 \
    --gpu_id 0 \
    --nc 3
~~~
If you want to test the reconstruction error of the image, the above settings will work.<br>
If you want to test the denoising of the image, set "test_in_dir" to "directory of noisy images" and "test_out_dir" to "directory of output ground truth".<br>
However, the two file names must correspond.

### 4.1. Training 2 (Top Level PixelSnail)

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/train2.sh
~~~
The following is an example of the training phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='celebA'

./VQ-VAE-2 \
    --train2 true \
    --train2_epochs 100 \
    --dataset ${DATA} \
    --size 256 \
    --train2_batch_size 16 \
    --gpu_id 0 \
    --nc 3
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/train2.sh
~~~

### 4.2. Test 2 (Top Level PixelSnail)

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/test2.sh
~~~
The following is an example of the test phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='celebA'

./VQ-VAE-2 \
    --test2 true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test2.sh
~~~

### 5.1. Training 3 (Bottom Level PixelSnail)

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/train3.sh
~~~
The following is an example of the training phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='celebA'

./VQ-VAE-2 \
    --train3 true \
    --train3_epochs 100 \
    --dataset ${DATA} \
    --size 256 \
    --train3_batch_size 16 \
    --gpu_id 0 \
    --nc 3
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/train3.sh
~~~

### 5.2. Test 3 (Bottom Level PixelSnail)

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/test3.sh
~~~
The following is an example of the test phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='celebA'

./VQ-VAE-2 \
    --test3 true \
    --dataset ${DATA} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test3.sh
~~~

### 6. Image Synthesis

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

./VQ-VAE-2 \
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

### 7. Image Sampling

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

./VQ-VAE-2 \
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


## Acknowledgments
This code is inspired by [vq-vae-2-pytorch](https://github.com/rosinality/vq-vae-2-pytorch).

