import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import mie_correction as mc

from scipy.io import loadmat


fig = plt.figure()

file = loadmat('/mnt/c/Users/conor/Google Drive/PhD/test_im.mat')

i=200
n=500

array = file['Image']
wn = file['WN'][0][i:n]

print(wn.shape, array.shape)

ref = array.mean(axis=0)[i:n]

app = array[np.random.randint(1,65000),i:n]


corr = mc.rmiesc(wn, app, ref, konevskikh=False)

print(corr)

plt.plot(wn, corr, label='corrected')
ax2=plt.gca().twinx()
ax2.plot(wn, app, label='raw',c='r')

plt.show()

fig.legend()
plt.savefig('miecorrected.png')