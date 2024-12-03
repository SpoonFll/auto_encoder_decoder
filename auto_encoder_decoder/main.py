#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig


def noise_gen(clean_encoding):
    noise = torch.randn(clean_encoding.size())
    return torch.add(clean_encoding,noise)

def main():
    q = 4
    bits = 1024
    m = int(bits/q)
    sig = torch.tensor(np.random.randint(2,size=bits)).to(torch.float32)
    print(sig)
    resig = torch.reshape(sig,(m,q))
    print(resig)
    lin = nn.Linear(q,q)
    linsig = lin(resig)
    print(linsig)


if __name__ == '__main__':
    main()
