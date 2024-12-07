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
    def forward(self,input_bits,B,Q,alpha):
        batch_size = input_bits.size(0)
        x = self.encoder(input_bits)
        x = power_constraint(x,B,Q)

        indices = torch.randint(0, M, (batch_size,)).to(device)  # Shape: (batch_size,)
        S_i = torch.nn.functional.one_hot(indices, num_classes=M).float().to(device)  # Shape: (batch_size, M)

        I = torch.sum(x * S_i, dim=1, keepdim=True)

        N_i = torch.randn(batch_size, 1).to(device)  # Shape: (batch_size, 1)
        I = I + N_i

        output_bits= self.decoder(I)
        Z_i=(torch.norm(output_bits)>=alpha).float().unsqueeze(1)
        return (output_bits,Z_i)


def power_constraint(X_batch, B, Q):
    power = torch.mean(torch.sum(X_batch ** 2, dim=1)) / Q
    if power > B:
        X_batch = X_batch * torch.sqrt(B / power)
    return X_batch

def main():
    num_bits = 1000

if __name__ == '__main__':
    main()
