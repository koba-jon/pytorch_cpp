# PatchCore
This is the implementation of "PatchCore".<br>
Original paper: K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler. Towards Total Recall in Industrial Anomaly Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. [link](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)

## Usage

### 0. Download pre-trained model
Please download ResNet pre-trained model with ImageNet.
~~~
$ wget https://huggingface.co/koba-jon/pre-train_cpp/resolve/main/models/resnet18.pth
$ wget https://huggingface.co/koba-jon/pre-train_cpp/resolve/main/models/wide_resnet50_2.pth
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
|    |--testN
|    |--testA
|    |--testGT
|
|--Dataset2
|--Dataset3
~~~


You should substitute the path of training normal data for "<training_path>", test normal data for "<test_normal_path>", test anomaly data for "<test_anomaly_path>", test ground truth data for "<test_gt_path>", respectively.<br>
The following is an example for "MVTecAD".
~~~
$ cd datasets
$ mkdir MVTecAD
$ cd MVTecAD
$ ln -s <training_path> ./train
$ ln -s <test_normal_path> ./testN
$ ln -s <test_anomaly_path> ./testA
$ ln -s <test_gt_path> ./testGT
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

./PatchCore \
    --train true \
    --dataset ${DATA} \
    --size 224 \
    --resnet_path "wide_resnet50_2.pth" \
    --n_layers "w50" \
    --coreset_rate 0.01 \
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

DATA='MVTecAD'

./PatchCore \
    --test true \
    --dataset ${DATA} \
    --size 224 \
    --resnet_path "wide_resnet50_2.pth" \
    --n_layers "w50" \
    --coreset_rate 0.01 \
    --gpu_id 0
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

./PatchCore \
    --AD true \
    --dataset ${DATA} \
    --normal_path "test_result/image_scoreN.txt" \
    --anomaly_path "test_result/image_scoreA.txt" \
    --AD_result_dir "AD_result/Image-AUROC"

./PatchCore \
    --AD true \
    --dataset ${DATA} \
    --normal_path "test_result/pixel_scoreN.txt" \
    --anomaly_path "test_result/pixel_scoreA.txt" \
    --AD_result_dir "AD_result/Pixel-AUROC"
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/anomaly_detection.sh
~~~


