import torch
import numpy as np
from scipy.stats import norm

class data_make():
    def __init__(x, num):
        self.x = x
        self.num = num
    def make_limit():
        y=len(self.x)
        p=[]
        if y>self.num:
            return self.x[-self.num:]
        elif y==self.num:
            return self.x
        else:
            z=[]
            z.extend(self.x)
            for i in range(self.num-y):
                z.append("")
            return z

    def age_limit():
        y=len(self.x)
        x=torch.tensor(self.x, dtype=torch.long)
        p=[]
        if y>self.num:
            x=x[-self.num:]
            p=x[0].clone()
            x-=p
            return x
        elif y==self.num:
            return x
        else:
            z=torch.cat((x,torch.zeros(self.num-y)))
            return z.to(torch.long)

    def make_padding():
        y=len(self.x)
        if y>=self.num:
            return torch.ones(self.num)
        elif y<self.num:
            return torch.cat((torch.ones(y),torch.zeros(self.num-y)))

def one_hot(label):
    z = torch.zeros(len(label),82)
    for i, j in enumerate(tqdm(label)):
        z[i,j] = 1
    return z

def MLDL(label,std=1,scaling=True):
    z = np.zeros((len(label),82))
    for i,j in enumerate(tqdm(label)):
        for t in j:
            z[i]=np.maximum(z[i],np.array(norm.pdf(range(82),loc=t,scale=std)))
        if scaling:
            z[i]=z[i]/max(z[i])
    z=torch.from_numpy(z).clone()
    return z