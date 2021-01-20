import pickle
import numpy as np
import os
import sys
import argparse
import urllib.request
import tarfile
from skimage import io

parser = argparse.ArgumentParser()

# Define parameter
parser.add_argument('--output1_dir', type=str)
parser.add_argument('--output2_dir', type=str)

args = parser.parse_args()


# Unpickle Dataset
def unpickle(file):
    f = open(file, 'rb')
    dict_dataset = pickle.load(f, encoding='latin1')
    f.close()
    return dict_dataset


# Get Dataset
def get_dataset(path):
    train_data = np.empty((0, 32*32*3))
    train_labels = np.empty(1)

    for i in range(1, 6):
        fname = os.path.join(path, f'data_batch_{i}')
        dict_dataset = unpickle(fname)
        if i == 1:
            train_data = dict_dataset['data']
            train_labels = dict_dataset['labels']
        else:
            train_data = np.vstack((train_data, dict_dataset['data']))
            train_labels = np.hstack((train_labels, dict_dataset['labels']))

    dict_dataset = unpickle(os.path.join(path, 'test_batch'))
    test_data = dict_dataset['data']
    test_labels = np.array(dict_dataset['labels'])

    bm = unpickle(os.path.join(path, 'batches.meta'))
    label_names = bm['label_names']
    return train_data, train_labels, test_data, test_labels, label_names


if __name__ == '__main__':

    # Download
    os.makedirs(f'{args.output1_dir}', exist_ok=False)
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    fname = url.split('/')[-1]
    fpath = os.path.join(args.output1_dir, fname)
    def _progress(cnt, chunk, total):
        now = cnt * chunk
        if now > total: now = total
        sys.stdout.write(f'\rdownloading {fname} {now} / {total} ({now/total:.1%})')
        sys.stdout.flush()
    urllib.request.urlretrieve(url, fpath, _progress)
    print('')

    # Unzip
    tarfile.open(fpath, 'r:gz').extractall(args.output1_dir)

    # Get Dataset
    train_data, train_labels, test_data, test_labels, label_names = get_dataset(f'{args.output1_dir}/cifar-10-batches-py')

    # Make Directories
    os.makedirs(f'{args.output2_dir}', exist_ok=False)
    os.makedirs(f'{args.output2_dir}/train', exist_ok=False)
    os.makedirs(f'{args.output2_dir}/test', exist_ok=False)
    for c in label_names:
        os.makedirs(f'{args.output2_dir}/train/{c}', exist_ok=False)
        os.makedirs(f'{args.output2_dir}/test/{c}', exist_ok=False)

    # Write Training Data
    train_len = len(train_data)
    train_count = [0] * train_len
    train_digit = 4
    for i in range(train_len):
        class_no = train_labels[i]
        image = np.rollaxis(train_data[i].reshape((3,32,32)), 0, 3)
        fname = f'{args.output2_dir}/train/{label_names[class_no]}/{str(train_count[class_no]).zfill(train_digit)}.png'
        io.imsave(fname, image)
        train_count[class_no] += 1

    # Write Test Data
    test_len = len(test_data)
    test_count = [0] * test_len
    test_digit = 3
    for i in range(test_len):
        class_no = test_labels[i]
        image = np.rollaxis(test_data[i].reshape((3,32,32)), 0, 3)
        fname = f'{args.output2_dir}/test/{label_names[class_no]}/{str(test_count[class_no]).zfill(test_digit)}.png'
        io.imsave(fname, image)
        test_count[class_no] += 1

