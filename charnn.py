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
        
        h = V(torch.zeros(in1.size()).cuda())
        h = F.tanh(self.l_hidden(h+in1))
        h = F.tanh(self.l_hidden(h+in2))
        h = F.tanh(self.l_hidden(h+in3))
        
        return F.log_softmax(self.l_out(h))


md = ColumnarModelData.from_arrays('.', [-1], np.stack([x1,x2,x3], axis=1), y, bs=512)

m = Char3Model(vocab_size, n_fac).cuda()

it = iter(md.trn_dl)
*xs,yt = next(it)
t = m(*V(xs))

opt = optim.Adam(m.parameters(), 1e-2)

fit(m, md, 1, opt, F.nll_loss)


