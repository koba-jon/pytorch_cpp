import os
import glob
import argparse
import xml.etree.ElementTree as ET


parser = argparse.ArgumentParser()

# Define parameter
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--class_list', type=str)

args = parser.parse_args()


# Set Class Names
def set_class_names(class_list):
    f = open(class_list, mode='r')
    class_names = []
    while True:
        line = f.readline().strip()
        if not line:
            break
        class_names += [line]
    f.close()
    return class_names


# Normalize Bounding Box
def normalizeBB(x_min, x_max, y_min, y_max, width, height):
    x_center = (x_min + x_max) * 0.5 / float(width)
    y_center = (y_min + y_max) * 0.5 / float(height)
    x_range = (x_max - x_min) / float(width)
    y_range = (y_max - y_min) / float(height)
    return x_center, y_center, x_range, y_range


# Convert XML into TXT
def convertXML2TXT(class_names, pathI, pathO):
    
    fileI = open(pathI, mode='r')
    fileO = open(pathO, mode='w')
    tree = ET.parse(fileI)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    for obj in root.iter('object'):
        class_name = obj.find('name').text
        class_id = class_names.index(class_name)
        BB = obj.find('bndbox')
        x_min = float(BB.find('xmin').text)
        x_max = float(BB.find('xmax').text)
        y_min = float(BB.find('ymin').text)
        y_max = float(BB.find('ymax').text)
        x_center, y_center, x_range, y_range = normalizeBB(x_min, x_max, y_min, y_max, width, height)
        fileO.write(f'{class_id} {x_center} {y_center} {x_range} {y_range}\n')
    
    fileI.close()
    fileO.close()


if __name__ == '__main__':

    # Get File Names
    fnames = []
    for f in glob.glob(f'{args.input_dir}/*.xml'):
        fnames.append(os.path.splitext(os.path.split(f)[1])[0])

    # Set Class Names
    class_names = set_class_names(args.class_list)

    # Convert XML into TXT
    os.makedirs(f'{args.output_dir}', exist_ok=False)
    for f in fnames:
        pathI = f'{args.input_dir}/{f}.xml'
        pathO = f'{args.output_dir}/{f}.txt'
        convertXML2TXT(class_names, pathI, pathO)
