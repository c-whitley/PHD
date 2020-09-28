import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import sys

sys.path.append('algorithms')

import mie_correction as mc

from scipy.io import loadmat
from Rubberband import Rubber_Band


fig = plt.figure()

file = loadmat('/mnt/c/Users/conor/Google Drive/PhD/test_im.mat')

i=200
n=500

array = file['Image']
wn = file['WN'][0][i:n]

ref = array.mean(axis=0)[i:n]

idx = [np.random.randint(1,65000) for _ in range(5)]

app = array[idx,i:n]

print(app.shape)


#corr = mc.rmiesc(wn, app, ref, konevskikh=False)

rb = Rubber_Band().fit_transform(app)

print(corr)

plt.plot(wn, corr, label='corrected')
ax2=plt.gca().twinx()
ax2.plot(wn, app, label='raw',c='r')

plt.show()

fig.legend()
plt.savefig('miecorrected.png')