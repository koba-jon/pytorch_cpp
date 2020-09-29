import os
import sys
import glob
import argparse
import random
from natsort import natsorted


parser = argparse.ArgumentParser()

# Define parameter
parser.add_argument('--input_dir1', type=str)
parser.add_argument('--input_dir2', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--output_label1', type=str)
parser.add_argument('--output_label2', type=str)
parser.add_argument('--k', type=int)
parser.add_argument('--seed_random', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()

files = []
for f in glob.glob(f'{args.input_dir1}/*'):
    files += [os.path.split(f)[1]]

fnames = []
for f in natsorted(files):
    fnames.append(f)
total = len(fnames)
valid_num = total/args.k

if args.seed_random==False:
    random.seed(args.seed)
random.shuffle(fnames)

input_abspath1 = os.path.abspath(args.input_dir1)
input_abspath2 = os.path.abspath(args.input_dir2)
os.makedirs(f'{args.output_dir}', exist_ok=False)
for j in range(args.k):
    os.makedirs(f'{args.output_dir}/{j+1}/train{args.output_label1}', exist_ok=False)
    os.makedirs(f'{args.output_dir}/{j+1}/train{args.output_label2}', exist_ok=False)
    os.makedirs(f'{args.output_dir}/{j+1}/valid{args.output_label1}', exist_ok=False)
    os.makedirs(f'{args.output_dir}/{j+1}/valid{args.output_label2}', exist_ok=False)
    for i, f in enumerate(fnames):
        pathI1 = f'{input_abspath1}/{f}'
        pathI2 = f'{input_abspath2}/{f}'
        if (i >= valid_num*j) and (i < valid_num*(j+1)):
            pathO1 = f'{args.output_dir}/{j+1}/valid{args.output_label1}/{f}'
            pathO2 = f'{args.output_dir}/{j+1}/valid{args.output_label2}/{f}'
        else:
            pathO1 = f'{args.output_dir}/{j+1}/train{args.output_label1}/{f}'
            pathO2 = f'{args.output_dir}/{j+1}/train{args.output_label2}/{f}'
        os.symlink(pathI1, pathO1)
        os.symlink(pathI2, pathO2)
