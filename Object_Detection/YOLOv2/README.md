# YOLOv2
This is the implementation of "YOLOv2" for Object Detection.<br>
Original paper: J. Redmon and A. Farhadi. YOLO9000: Better, Faster, Stronger. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. [link](https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)

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
!/bin/bash

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
!/bin/bash

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

./YOLOv2 \
    --train true \
    --augmentation true \
    --epochs 300 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 608 \
    --batch_size 8 \
    --prob_thresh 0.03 \
    --Lambda_noobject 0.1 \
    --lr_init 1e-5 \
    --lr_base 1e-4 \
    --lr_decay1 1e-5 \
    --lr_decay2 1e-6 \
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

./YOLOv2 \
    --test true \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 608 \
    --prob_thresh 0.03 \
    --Lambda_noobject 0.1 \
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

./YOLOv2 \
    --detect true \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 608 \
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

./YOLOv2 \
    --demo true \
    --cam_num 0 \
    --dataset ${DATA} \
    --class_list "list/${DATA}.txt" \
    --class_num 20 \
    --size 608 \
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
This code is inspired by [darknet](https://github.com/pjreddie/darknet), [Yolo-v2-pytorch](https://github.com/uvipen/Yolo-v2-pytorch), [yolov2.pytorch](https://github.com/tztztztztz/yolov2.pytorch), [yolo2-pytorch](https://github.com/longcw/yolo2-pytorch), [furkanu/yolov2-pytorch](https://github.com/furkanu/yolov2-pytorch), [yxlijun/yolov2-pytorch](https://github.com/yxlijun/yolov2-pytorch) and [YOLOv2](https://github.com/leetenki/YOLOv2).


---

---

---

## Training Strategy

---

### Loss

Loss function (default):
![YOLOv2_loss](https://user-images.githubusercontent.com/56967584/111178592-ab71ba80-85ee-11eb-99e7-1abe2c414b60.png)

If the loss of term `conf<noobj>` is strong, "Not Detected" will occur frequently. <br>
In the case, it is recommended to add `--Lambda_noobject 0.1` to arguments, where the default value is `1.0`.<br>

---

### Learning Rate

If the initial value of `learning rate` is high, gradient values will diverge. <br>
In the case, it is recommended to add `--lr_init 1e-5`, `--lr_base 1e-4`, `--lr_decay1 1e-5` and `--lr_decay2 1e-6` to arguments. (i.e., slow updation and fluctuation)<br>

---

### Data Augmentation

Transformation of 8 components:
- Flipping : `--flip_rate 0.5`
- Scaling (i.e., Resize) : `--scale_rate 0.5`
- Blurring (i.e., Applying an averaging filter) : `--blur_rate 0.5`
- Change Brightness (i.e., Value in HSV) : `--brightness_rate 0.5`
- Change Hue : `--hue_rate 0.5`
- Change Saturation : `--saturation_rate 0.5`
- Shifting : `--shift_rate 0.5`
- Cropping : `--crop_rate 0.5`

Please write the occurrence probability in each argument `rate`.<br>
Here, `1.0` means that it always occurs.<br>

If the distributions of training and test samples are quite similar, data augmentation has undesirable effect.<br>
In the case, it is recommended to add `--augmentation false` to arguments, where the default value is `true`.<br>

---

### Anchor

Anchors are useful for stable detection.<br>
Please refer to `cfg/anchor.txt` to change the config.<br>

~~~
0.57273 0.667385    # (1) Prior-width Prior-height
1.87446 2.06253     # (2) Prior-width Prior-height
3.33843 5.47434     # (3) Prior-width Prior-height
7.88282 3.52778     # (4) Prior-width Prior-height
9.77052 9.16828     # (5) Prior-width Prior-height
~~~

Please write the prior-size, where each grid size is `1.0`.<br>
If you want to change types of anchor, please write `types` on `--na` to arguments.

---

### Resize

Multi-Scale Training allows the predictor to detect objects at various resolutions.<br>
Please refer to `cfg/resize.txt` to change the config.<br>

~~~
10          # types of image size to resize (i.e., it must match the number of lines for size in the text file.)
10          # iterations to switch
320 320     # (1) Width Height
352 352     # (2) Width Height
384 384     # (3) Width Height
416 416     # (4) Width Height
448 448     # (5) Width Height
480 480     # (6) Width Height
512 512     # (7) Width Height
544 544     # (8) Width Height
576 576     # (9) Width Height
608 608     # (10) Width Height
~~~

---

---

## Detection Strategy

Detection performance is determined by the prediction result and two threshold.

Two threshold:
- Simultaneous probability with confidence and class score : `--prob_thresh 0.1`
- IoU between bounding boxes in Non-Maximum Suppression : `--nms_thresh 0.5`

If you allow over-detection, please decrease `prob_thresh` and increase `nms_thresh`. (e.g., `--prob_thresh 0.05`, `--nms_thresh 0.75`)
