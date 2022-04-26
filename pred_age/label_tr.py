import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch

def MLDL(label, size, std=1,scaling=True):
    z = np.zeros((len(label),size))
    for i,j in enumerate(tqdm(label)):
        for t in j:
            z[i]=np.maximum(z[i],np.array(norm.pdf(range(size),loc=t,scale=std)))
        if scaling:
            z[i]=z[i]/max(z[i])
    z=torch.from_numpy(z).clone()
    return z

def one_hot(label, size):
    onehot = torch.zeros(len(label),size)
    for i, j in enumerate(tqdm(label)):
        onehot[i,j] = 1
    return onehot