#!/usr/bin/env python3
import sys
sys.path.append('../build/lib')
import matplotlib.pylab as plt
from scipy.io import mmread,mmwrite
import pyms
import numpy as np

surfix=['1','2']

ra=[]
dec=[]
vis=[]
noise=[]
for s in surfix:
    ra.append(mmread('ra'+s)[:,0])
    dec.append(mmread('dec'+s)[:,0])
    vis.append(mmread('vis'+s)[:,0])
    noise.append(np.zeros_like(vis[-1]))
    noise[-1][0]=1.0


(p,m)=pyms.define_pixels_mo(ra, dec, 1)

####
s=pyms.brute_solver_mo(p, vis, noise)
solution=pyms.solve(s)
####
pyms.plot_hit_map_mo(p, m)
plt.show()
img=pyms.fill_map(solution, m)
#plt.figure(figsize=(10,10))
plt.imshow(img, aspect='auto')
plt.colorbar()
plt.show()
