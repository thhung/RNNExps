#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This script is an example of implementation of 3-char model RNN in unrolled format.
"""
import os
import sys

import torch
from torch import utils
import torch.utils.data as utils
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

## prepare to data

# size of vocabulary using for this dataset
vocab_size = 0

# dictionary from char to indices
char_indices = None

# list of chars in the chosen dataset
chars = None


class CustomDataset(utils.Dataset):
    """

        This class is an dataset class in pytorch which helps to easily manipulate
        the nietzsche dataset and return as vector. Please see how to use it below.
    """

    def __init__(self, data_path=None):
        # TODO
        # 1. Initialize file path or list of file names.
        # place to save dataset
        data_path = 'data/nietzsche/'

        # this function below doesn't work as expected.
        # do it manually instead.
        #get_data("https://s3.amazonaws.com/text-datasets/nietzsche.txt", PATH + 'nietzsche.txt')
        text = open(data_path + 'nietzsche.txt').read()

        #get sorted list of all chars in this text
        global chars
        chars = sorted(list(set(text)))
        global vocab_size
        vocab_size = len(chars) + 1

        # add "\0" for padding
        chars.insert(0, "\0")

        #global so we can use it later.
        global char_indices
        char_indices = {c: i for i, c in enumerate(chars)}
        # just in case we want map from indices to char
        indices_char = {i: c for i, c in enumerate(chars)}

        idx = [char_indices[c] for c in text]

        # self.cs is number of chars we would like to use as input
        # in this case, of course it is 3 but this variable help if we would
        # like to change it to 8, for example.
        self.cs = 3

        # self.c[number]_dat is the indices of chars we get from dataset.
        # It also serves as input after we concat them.
        self.c1_dat = [idx[i] for i in range(0, len(idx) - self.cs, self.cs)]
        self.c2_dat = [
            idx[i + 1] for i in range(0,
                                      len(idx) - self.cs, self.cs)
        ]
        self.c3_dat = [
            idx[i + 2] for i in range(0,
                                      len(idx) - self.cs, self.cs)
        ]
        self.c4_dat = [
            idx[i + 3] for i in range(0,
                                      len(idx) - self.cs, self.cs)
        ]

    def __getitem__(self, index):
        # Simply return the item we want. Thanks for preprocess in __init__()
        # What if data doesn't fit?
        return (self.c1_dat[index], self.c2_dat[index],
                self.c3_dat[index]), self.c4_dat[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.c4_dat)


#######################################################################
## How to use my CustomDataset
#######################################################################

custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(
    dataset=custom_dataset, batch_size=100, shuffle=True, num_workers=2)

#######################################################################

# These two variable below will use for Char3Model
n_hidden = 256
n_fac = 42


class Char3Model(nn.Module):
    """

        This neural net uses three chars as the inputs and predict the char after them
        by using a simple RNN model with 1 hidden layers (1 hidden layer here is in context
        of RNN, shouldn't confuse with normal neural net in my implementation, it should be equivalent
        to 3 layers).
    """

    def __init__(self, vocab_size, n_fac):
        """
            vocab_size : the size of vocabulary (dictionary)
            n_fac      : size of embedding vector
            n_hidden   : number of neural in hidden layer
        """
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.l_in = nn.Linear(n_fac, n_hidden)
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)

    def forward(self, c1, c2, c3):
        in1 = F.relu(self.l_in(self.e(c1)))
        in2 = F.relu(self.l_in(self.e(c2)))
        in3 = F.relu(self.l_in(self.e(c3)))

        h = Variable(torch.zeros(in1.size()).cuda())
        h = F.tanh(self.l_hidden(h + in1))
        h = F.tanh(self.l_hidden(h + in2))
        h = F.tanh(self.l_hidden(h + in3))

        return F.log_softmax(self.l_out(h))


# create model and move it to GPU
m = Char3Model(vocab_size, n_fac).cuda()

#prepare for trainning
opt = optim.Adam(m.parameters(), 1e-2)
#set number of epochs
nb_epochs = 2

# start training
for epoch in range(nb_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        i1, i2, i3 = inputs[0], inputs[1], inputs[2]

        i1, i2, i3, labels = Variable(torch.from_numpy(
            np.array(i1))), Variable(torch.from_numpy(np.array(i2))), Variable(
                torch.from_numpy(np.array(i3))), Variable(labels)

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        i1 = i1.cuda()
        i2 = i2.cuda()
        i3 = i3.cuda()
        labels = labels.cuda()
        outputs = m(i1, i2, i3)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        opt.step()

        # print statistics

        running_loss += loss.data[0]
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,
                                            running_loss / 2000.0))
            running_loss = 0.0

print("Finished training")


def get_next(inp):
    """
        This function take the input and return the predicted char after the chars in input
        Arg:
            inp: 3-chars string
        Return:
            Predicted char
    """
    idxs = torch.Tensor(np.array([char_indices[c] for c in inp]))
    mc_type = torch.cuda.LongTensor
    x1, x2, x3 = [
        Variable(torch.from_numpy(np.array([o])).type(mc_type)) for o in idxs
    ]

    p = m(x1, x2, x3)
    i = np.argmax(p.data)
    return chars[i]


print("Predict of |y. | is ", get_next('y. '))
print("Predict of |peo| is ", get_next('peo'))
print("Predict of |han| is ", get_next('han'))
