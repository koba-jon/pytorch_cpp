# U-Net Classification
This is the implementation of "U-Net" for classification.<br>
Original paper: O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 2015. [link](https://arxiv.org/abs/1505.04597)

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
|    |    |--label1.png
|    |    |--label2.png
|    |    |--label3.png
|    |
|    |--validI
|    |--validO
|    |--testI
|    |--testO
|
|--Dataset2
|--Dataset3
~~~

You should substitute the path of training input data for "<training_input_path>", training output data for "<training_output_path>", test input data for "<test_input_path>", test output data for "<test_output_path>", respectively.<br>
The following is an example for "VOC2012".
~~~
$ cd datasets
$ mkdir VOC2012
$ cd VOC2012
$ ln -s <training_input_path> ./trainI
$ ln -s <training_output_path> ./trainO
$ ln -s <test_input_path> ./testI
$ ln -s <test_output_path> ./testO
$ cd ../..
~~~

#### Change Class Label
You can change the class label as follows.<br>
The following is an example for "VOC2012", and you can change the index value from 255 to 21.
~~~
$ vi src/main.cpp
~~~

Please comment out "transforms::ConvertIndex" of "transformO".
~~~
std::vector<transforms::Compose*> transformO{
    (transforms::Compose*)new transforms::Resize(cv::Size(vm["size"].as<size_t>(), vm["size"].as<size_t>()), cv::INTER_NEAREST),
    (transforms::Compose*)new transforms::ConvertIndex(255, 21),
    (transforms::Compose*)new transforms::ToTensorLabel()
};
~~~

If you change the code, you need to build the code again.
~~~
$ cd build
$ make -j4
$ cd ..
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

./U-Net_Classification \
    --train true \
    --epochs 300 \
    --dataset ${DATA} \
    --class_num 22 \
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

DATA='VOC2012'

./U-Net_Classification \
    --test true \
    --dataset ${DATA} \
    --class_num 22 \
    --size 256 \
    --gpu_id 0 \
    --nc 3
~~~
The ground truth of output label in this network must be a png image in index color format, where index value in each pixel corresponds to a class label.<br>
In addition, the input image in this network is not particular about.<br>
However, the two file names must correspond without the extension.

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/test.sh
~~~


## Acknowledgments
This code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
