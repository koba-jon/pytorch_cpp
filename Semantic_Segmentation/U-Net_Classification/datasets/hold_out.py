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
parser.add_argument('--train_rate', type=float)
parser.add_argument('--valid_rate', type=float)
parser.add_argument('--seed_random', action='store_true', help='whether to make the seed of random number in a random')
parser.add_argument('--seed', type=int, default=0, help='the seed of random number')

args = parser.parse_args()

files = []
for f in glob.glob(f'{args.input_dir}/*'):
    files += [os.path.split(f)[1]]

fnames = []
for f in natsorted(files):
    fnames.append(f)
total = len(fnames)
valid_num = int(total * args.valid_rate / (args.train_rate + args.valid_rate))

if not args.seed_random:
    random.seed(args.seed)
random.shuffle(fnames)

input_abspath = os.path.abspath(args.input_dir)
os.makedirs(f'{args.output_dir}', exist_ok=False)
os.makedirs(f'{args.output_dir}/train', exist_ok=False)
os.makedirs(f'{args.output_dir}/valid', exist_ok=False)
for i, f in enumerate(fnames):
    pathI = f'{input_abspath}/{f}'
    if i < valid_num:
        pathO = f'{args.output_dir}/valid/{f}'
    else:
        pathO = f'{args.output_dir}/train/{f}'
    os.symlink(pathI, pathO)
