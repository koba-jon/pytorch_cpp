import os
import sys
import glob
import argparse
import random
from natsort import natsorted


parser = argparse.ArgumentParser()

# Define parameter
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--k', type=int)
parser.add_argument('--seed_random', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()

files = []
for f in glob.glob(f'{args.input_dir}/*'):
    files += [os.path.split(f)[1]]

fnames = []
for f in natsorted(files):
    fnames.append(f)
total = len(fnames)
valid_num = total/args.k

if args.seed_random==False:
    random.seed(args.seed)
random.shuffle(fnames)

input_abspath = os.path.abspath(args.input_dir)
os.makedirs(f'{args.output_dir}', exist_ok=False)
for j in range(args.k):
    os.makedirs(f'{args.output_dir}/{j+1}/train', exist_ok=False)
    os.makedirs(f'{args.output_dir}/{j+1}/valid', exist_ok=False)
    for i, f in enumerate(fnames):
        pathI = f'{input_abspath}/{f}'
        if (i >= valid_num*j) and (i < valid_num*(j+1)):
            pathO = f'{args.output_dir}/{j+1}/valid/{f}'
        else:
            pathO = f'{args.output_dir}/{j+1}/train/{f}'
        os.symlink(pathI, pathO)
