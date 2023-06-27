# EGBAD2d
This is the implementation of "EGBAD" corresponding to 2-dimensional shape.<br>
Original paper: H. Zenati, C. S. Foo, B. Lecouat, G. Manek, and V. R. Chandrasekhar. Efficient gan-based anomaly detection. In the Workshop of International Conference on Learning Representations, 2018. [link](https://arxiv.org/abs/1802.06222)

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
- MVTec Anomaly Detection Dataset (MVTec AD)<br>
This is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection.<br>
Link: [official](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

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
|    |--test_anomaly
|    |--test_normal
|
|--Dataset2
|--Dataset3
~~~


You should substitute the path of training normal data for "<training_path>", test anomaly data for "<test_anomaly_path>", test normal data for "<test_normal_path>", respectively.<br>
The following is an example for "MVTecAD".
~~~
$ cd datasets
$ mkdir MVTecAD
$ cd MVTecAD
$ ln -s <training_path> ./train
$ ln -s <test_anomaly_path> ./test_anomaly
$ ln -s <test_normal_path> ./test_normal
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

DATA='MVTecAD'

./EGBAD2d \
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

DATA='MVTecAD'
HEATMAP_MAX=0.1

./EGBAD2d \
    --test true \
    --dataset ${DATA} \
    --test_dir "test_anomaly" \
    --test_result_dir "test_result_anomaly" \
    --heatmap_max ${HEATMAP_MAX} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
    
./EGBAD2d \
    --test true \
    --dataset ${DATA} \
    --test_dir "test_normal" \
    --test_result_dir "test_result_normal" \
    --heatmap_max ${HEATMAP_MAX} \
    --size 256 \
    --gpu_id 0 \
    --nc 3
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~

### 5. Anomaly Detection

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/anomaly_detection.sh
~~~
The following is an example of the anomaly detection phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='MVTecAD'

./EGBAD2d \
    --AD true \
    --dataset ${DATA} \
    --anomaly_path "test_result_anomaly/anomaly_score.txt" \
    --normal_path "test_result_normal/anomaly_score.txt" \
    --n_thresh 256
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/anomaly_detection.sh
~~~


## Acknowledgments
This code is inspired by [pytorch_advanced](https://github.com/YutaroOgawa/pytorch_advanced).
