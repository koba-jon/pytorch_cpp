import numpy as np
import os
import sys
import argparse
import urllib.request
import gzip

parser = argparse.ArgumentParser()

# Define parameter
parser.add_argument('--output1_dir', type=str)
parser.add_argument('--output2_dir', type=str)

args = parser.parse_args()

url_base = 'http://yann.lecun.com/exdb/mnist/'
fnames = {
    'train_data':'train-images-idx3-ubyte.gz',
    'train_labels':'train-labels-idx1-ubyte.gz',
    'test_data':'t10k-images-idx3-ubyte.gz',
    'test_labels':'t10k-labels-idx1-ubyte.gz'
}


# Load Label
def load_labels(path):
    with gzip.open(path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels


# Load Image
def load_data(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    return data


# Get Dataset
def get_dataset(path):
    train_data = load_data(path + '/' + fnames['train_data'])
    train_labels = load_labels(path + '/' + fnames['train_labels'])
    test_data = load_data(path + '/' + fnames['test_data'])
    test_labels = load_labels(path + '/' + fnames['test_labels'])
    label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return train_data, train_labels, test_data, test_labels, label_names


if __name__ == '__main__':

    # Download
    os.makedirs(f'{args.output1_dir}', exist_ok=False)
    for _, fname in fnames.items():
        fpath = os.path.join(args.output1_dir, fname)
        def _progress(cnt, chunk, total):
            now = cnt * chunk
            if now > total: now = total
            sys.stdout.write(f'\rdownloading {fname} {now} / {total} ({now/total:.1%})')
            sys.stdout.flush()
        urllib.request.urlretrieve(url_base+fname, fpath, _progress)
        print('')

    # Get Dataset
    train_data, train_labels, test_data, test_labels, label_names = get_dataset(args.output1_dir)

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
        fname = f'{args.output2_dir}/train/{label_names[class_no]}/{str(train_count[class_no]).zfill(train_digit)}.dat'
        np.savetxt(fname, train_data[i])
        train_count[class_no] += 1

    # Write Test Data
    test_len = len(test_data)
    test_count = [0] * test_len
    test_digit = 4
    for i in range(test_len):
        class_no = test_labels[i]
        fname = f'{args.output2_dir}/test/{label_names[class_no]}/{str(test_count[class_no]).zfill(test_digit)}.dat'
        np.savetxt(fname, test_data[i])
        test_count[class_no] += 1

