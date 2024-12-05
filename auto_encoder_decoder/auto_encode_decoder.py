import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Parameters
Q = 4
M = 6
B = 1
samples = 1000
variance = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def noise_gen(clean_encoding):
    noise = torch.randn(clean_encoding.sizze())
    return clean_encoding + noise

class Auto_Encoder(nn.Module):
    def __init__(self, M):
        super(Auto_Encoder, self).__init__()
        
        #Encoder
        self.encoder = nn.Sequential(
            nn.Linear(M, 64),
            nn.ReLU(),
            nn.Linear(M, 64),
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(M, 64),
            nn.ReLU(),
            nn.Linear(M, 64),
        )
def main():
    
if __name__ == '__main__':
    main()