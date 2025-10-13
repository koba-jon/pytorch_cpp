# YOLOv5
This is the implementation of "YOLOv5" for Object Detection.<br>
Reference: [link](https://github.com/ultralytics/yolov5)

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

./YOLOv5 \
    --train true \
    --augmentation true \
    --model "yolov5s" \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 640 \
    --batch_size 16 \
    --prob_thresh 0.03 \
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

./YOLOv5 \
    --test true \
    --model "yolov5s" \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 640 \
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

./YOLOv5 \
    --detect true \
    --model "yolov5s" \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 640 \
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

./YOLOv5 \
    --demo true \
    --cam_num 0 \
    --model "yolov5s" \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 640 \
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
This code is inspired by [yolov5](https://github.com/ultralytics/yolov5/tree/master).


---

---

---

## Training Strategy

---

### Data Augmentation

Transformation of 10 components:
- Flipping : `--flip_rate 0.5`
- Scaling (i.e., Resize) : `--scale_rate 0.5`
- Blurring (i.e., Applying an averaging filter) : `--blur_rate 0.5`
- Change Brightness (i.e., Value in HSV) : `--brightness_rate 0.5`
- Change Hue : `--hue_rate 0.5`
- Change Saturation : `--saturation_rate 0.5`
- Shifting : `--shift_rate 0.5`
- Cropping : `--crop_rate 0.5`
- Mosaic : `--mosaic_rate 0.5`
- Mixup : `--mixup_rate 0.5`

Please write the occurrence probability in each argument `rate`.<br>
Here, `1.0` means that it always occurs.<br>

If the distributions of training and test samples are quite similar, data augmentation has undesirable effect.<br>
In the case, it is recommended to add `--augmentation false` to arguments, where the default value is `true`.<br>

---

### Anchor

Anchors are useful for stable detection.<br>
Please refer to `cfg/anchor.txt` to change the config.<br>

~~~
10.0 13.0     # (scale1.anchor1) Prior-width Prior-height
16.0 30.0     # (scale1.anchor2) Prior-width Prior-height
33.0 23.0     # (scale1.anchor3) Prior-width Prior-height
30.0 61.0     # (scale2.anchor1) Prior-width Prior-height
62.0 45.0     # (scale2.anchor2) Prior-width Prior-height
59.0 119.0    # (scale2.anchor3) Prior-width Prior-height
116.0 90.0    # (scale3.anchor1) Prior-width Prior-height
156.0 198.0   # (scale3.anchor2) Prior-width Prior-height
373.0 326.0   # (scale3.anchor3) Prior-width Prior-height
~~~

Please write the prior-size, where each pixel size is `1.0` on image.<br>
If you want to change types of anchor, please write `types` on `--na` to arguments.

---


## Detection Strategy

Detection performance is determined by the prediction result and two threshold.

Two threshold:
- Simultaneous probability with confidence and class score : `--prob_thresh 0.1`
- IoU between bounding boxes in Non-Maximum Suppression : `--nms_thresh 0.5`

If you allow over-detection, please decrease `prob_thresh` and increase `nms_thresh`. (e.g., `--prob_thresh 0.05`, `--nms_thresh 0.75`)
