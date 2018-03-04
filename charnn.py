import sys

import os
import torch
from torch import utils
import torch.utils.data as utils
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

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


tensor_x = torch.LongTensor((c1_dat, c2_dat, c3_dat))
tensor_x = torch.t(tensor_x)
t_y = torch.LongTensor(c4_dat)
tensor_y = t_y #torch.t(t_y)

my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
trainloader = utils.DataLoader(my_dataset) # create your dataloader

y = np.stack(c4_dat)

n_hidden = 256

n_fac = 42

class Char3Model(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)

        # The 'green arrow' from our diagram - the layer operation from input to hidden
        self.l_in = nn.Linear(n_fac, n_hidden)

        # The 'orange arrow' from our diagram - the layer operation from hidden to hidden
        self.l_hidden = nn.Linear(n_hidden, n_hidden)

        # The 'blue arrow' from our diagram - the layer operation from hidden to output
        self.l_out = nn.Linear(n_hidden, vocab_size)

    def forward(self, c1, c2, c3):
        in1 = F.relu(self.l_in(self.e(c1)))
        in2 = F.relu(self.l_in(self.e(c2)))
        in3 = F.relu(self.l_in(self.e(c3)))

        h = Variable(torch.zeros(in1.size()).cuda())
        h = F.tanh(self.l_hidden(h+in1))
        h = F.tanh(self.l_hidden(h+in2))
        h = F.tanh(self.l_hidden(h+in3))

        return F.log_softmax(self.l_out(h))


m = Char3Model(vocab_size, n_fac).cuda()

opt = optim.Adam(m.parameters(), 1e-2)

#set number of epochs
nb_epochs = 2

#criterion = F.nll_loss()

for epoch in range(nb_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        #print(inputs)
        i1, i2, i3 = inputs[0,0], inputs[0,1], inputs[0,2]

        i1, i2, i3, labels = Variable(torch.from_numpy(np.array([i1]))), Variable(torch.from_numpy(np.array([i2]))), Variable(torch.from_numpy(np.array([i3]))), Variable(labels)

        # zero the parameter gradients
        opt.zero_grad()

        #print(i1, i2, i3)
        # forward + backward + optimize
        i1 = i1.cuda()
        i2 = i2.cuda()
        i3 = i3.cuda()
        labels = labels.cuda()
        outputs = m(i1, i2, i3)
        #print(outputs, labels)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        opt.step()

        # print statistics

        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000.0))
            running_loss = 0.0
        #if i == 10:
        #    break
    #break
print("Finished training")
