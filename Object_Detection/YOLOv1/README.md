# YOLOv1
This is the implementation of "YOLOv1" for Object Detection.<br>
Original paper: J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016. [link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)

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
- The PASCAL Visual Object Classes Challenge 2012 (VOC2012)<br>
This is a set of images that has an annotation file giving a bounding box and object class label for each object in one of the twenty classes present in the image.<br>
Link: [official](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

#### Setting

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
|    |--trainO
|    |    |--label1.txt
|    |    |--label2.txt
|    |    |--label3.txt
|    |
|    |--validI
|    |--validO
|    |--testI
|    |--testO
|    |
|    |--detect
|         |--image4.png
|         |--image5.bmp
|         |--image6.jpg
|
|--Dataset2
|--Dataset3
~~~

- Class List

Please set the text file for class names.
~~~
$ vi list/VOC2012.txt
~~~

In case of "VOC2012", please set as follows.
~~~
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
~~~

- Input Image

You should substitute the path of training input data for "<training_input_path>", test input data for "<test_input_path>", detection input data for "<detect_path>", respectively.<br>
The following is an example for "VOC2012".
~~~
$ cd datasets
$ mkdir VOC2012
$ cd VOC2012
$ ln -s <training_input_path> ./trainI
$ ln -s <test_input_path> ./testI
$ ln -s <detect_path> ./detect
$ cd ../..
~~~

- Output Label

You should get the id (class number), x-coordinate center, y-coordinate center, width, and height from the class and coordinate data of bounding boxes in the XML file and normalize them.<br>
Please follow the steps below to convert XML file to text file.<br>
Here, you should substitute the path of training XML data for "<training_xml_path>", test XML data for "<test_xml_path>", respectively.<br>
The following is an example for "VOC2012".
~~~
$ ln -s <training_xml_path> ./datasets/VOC2012/trainX
$ ln -s <test_xml_path> ./datasets/VOC2012/testX
~~~

Please create a text file for training data.
~~~
$ vi ../../scripts/xml2txt.sh
~~~

You should substitute the path of training XML data for "--input_dir", training text data for "--output_dir", class name list for "--class_list", respectively.
~~~
#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

python3 ${SCRIPT_DIR}/xml2txt.py \
    --input_dir "datasets/VOC2012/trainX" \
    --output_dir "datasets/VOC2012/trainO" \
    --class_list "list/VOC2012.txt"
~~~

The data will be converted by the following procedure.
~~~
$ sh ../../scripts/xml2txt.sh
~~~

Please create a text file for test data.
~~~
$ vi ../../scripts/xml2txt.sh
~~~

You should substitute the path of test XML data for "--input_dir", test text data for "--output_dir", class name list for "--class_list", respectively.
~~~
#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

python3 ${SCRIPT_DIR}/xml2txt.py \
    --input_dir "datasets/VOC2012/testX" \
    --output_dir "datasets/VOC2012/testO" \
    --class_list "list/VOC2012.txt"
~~~

The data will be converted by the following procedure.
~~~
$ sh ../../scripts/xml2txt.sh
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

DATA='VOC2012'

./YOLOv1 \
    --train true \
    --augmentation true \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 448 \
    --batch_size 16 \
    --prob_thresh 0.03 \
    --lr_init 1e-4 \
    --lr_base 1e-3 \
    --lr_decay1 1e-4 \
    --lr_decay2 1e-5 \
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

DATA='VOC2012'

./YOLOv1 \
    --test true \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 448 \
    --prob_thresh 0.03 \
    --gpu_id 0 \
    --nc 3
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~

### 5. Detection

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/detect.sh
~~~
The following is an example of the detection phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='VOC2012'

./YOLOv1 \
    --detect true \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 448 \
    --prob_thresh 0.03 \
    --gpu_id 0 \
    --nc 3
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/detect.sh
~~~

### 6. Demo

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/demo.sh
~~~
The following is an example of the demo phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='VOC2012'

./YOLOv1 \
    --demo true \
    --cam_num 0 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 448 \
    --prob_thresh 0.03 \
    --gpu_id 0 \
    --nc 3
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/demo.sh
~~~


## Acknowledgments
This code is inspired by [darknet](https://github.com/pjreddie/darknet), [yolo_v1_pytorch](https://github.com/motokimura/yolo_v1_pytorch), and [pytorch_yolov1](https://github.com/Tshzzz/pytorch_yolov1).

