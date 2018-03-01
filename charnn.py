import os
import torch
from fastai.io import *
from fastai.conv_learner import *

from fastai.column_data import *

## prepare to data

# place to save dataset
PATH='data/nietzsche/'

# this function below doesn't work as expected. 
# do it manually instead. 
#get_data("https://s3.amazonaws.com/text-datasets/nietzsche.txt", PATH + 'nietzsche.txt')
text = open(PATH + 'nietzsche.txt').read()

#get sorted list of all chars in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)+1

# add "\0" for padding
chars.insert(0, "\0")

char_indices = {c: i for i, c in enumerate(chars)}
indices_char = {i: c for i, c in enumerate(chars)}

idx = [char_indices[c] for c in text]

cs=3
c1_dat = [idx[i]   for i in range(0, len(idx)-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-cs, cs)]
c4_dat = [idx[i+3] for i in range(0, len(idx)-cs, cs)]

x1 = np.stack(c1_dat)
x2 = np.stack(c2_dat)
x3 = np.stack(c3_dat)

y = np.stack(c4_dat)

