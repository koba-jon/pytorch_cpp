# NeRF
This is the implementation of "NeRF".<br>
Original paper: B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In European Conference on Computer Vision, 2020. [link](https://arxiv.org/abs/2003.08934)

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
- srn cars Dataset / srn chairs Dataset<br>
Link: [official](https://github.com/sxyu/pixel-nerf?tab=readme-ov-file)

#### Setting

Please prepare RGB or grayscale images that can be read by OpenCV, and 16-dimensional (flattened 4x4 matrix) camera pose parameters in text files.
Also, all data must be paired images with matching file name excluding extension.

- Example: 000000.txt
~~~
0.6811988353729248 -0.6771866083145142 -0.278184175491333 0.3616396188735962 0.732098400592804 0.6301046013832092 0.25884315371513367 -0.3364964723587036 0.0 -0.3799818456172943 0.9249937534332275 -1.2024919986724854 -0.0 0.0 -0.0 1.0
~~~

Please create a link for the dataset.<br>
The following hierarchical relationships are recommended.

~~~
datasets
|--Dataset1
|    |--trainI
|    |    |--image1.png
|    |    |--image2.bmp
|    |    |--image3.jpg
|    |
|    |--trainP
|    |    |--image1.txt
|    |    |--image2.txt
|    |    |--image3.txt
|    |
|    |--validI
|    |--validP
|    |--testI
|    |--testP
|
|--Dataset2
|--Dataset3
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

DATA='srn_cars'

./NeRF \
    --train true \
    --train_load_epoch "latest" \
    --epochs 10000 \
    --dataset ${DATA} \
    --size 128 \
    --focal_length 131.25 \
    --batch_size 1 \
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

DATA='srn_cars'

./NeRF \
    --test true \
    --dataset ${DATA} \
    --size 128 \
    --focal_length 131.25 \
    --gpu_id 0
~~~
If you want to test the reconstruction error of the image, the above settings will work.<br>
If you want to test the denoising of the image, set "test_in_dir" to "directory of noisy images" and "test_out_dir" to "directory of output ground truth".<br>
However, the two file names must correspond.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~

### 5. Image Sampling

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/sample.sh
~~~
The following is an example of the sampling phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='srn_cars'

./NeRF \
    --sample true \
    --dataset ${DATA} \
    --sample_total 100 \
    --size 128 \
    --focal_length 131.25 \
    --gpu_id 0
~~~
If you want to generate image, the above settings will work.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/sample.sh
~~~

