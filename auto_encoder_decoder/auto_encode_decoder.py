import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Parameters
Q = 4
q = 2
M = 6
B = 1
samples = 1000
variance = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Auto_Encoder(nn.Module):
    def __init__(self, M):
        super(Auto_Encoder, self).__init__()
        
        #Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1000, q),
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(q, 1000),
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

def evaluate(B_values, Q_values, alpha_values):
    BER_results = {1: 0.1, 2: 0.2, 3: 0.3}  # Dummy values
    BER = 0.1  # Dummy value

    for B in B_values:
        for Q in Q_values:
            for alpha in alpha_values:
                #BER = Simulate(B,Q, Alpah)
                BER_results[(B, Q, alpha)] = BER
                print(f'B={B}, Q={Q}, Alpha={alpha}, BER={BER}')
            
    return BER_results

def plot(BER_results, B_values, Q_values, alpha_values):
    plt.figure(figsize=(10, 6))

    for Q in Q_values:
        BER_plot = [BER_results[(B_values[0], Q, alpha)] for alpha in alpha_values]
        plt.plot(alpha_values, BER_plot, marker='o', label=f'Q={Q}')

    plt.xlabel('Alpha (Threshold)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs Alpha for Different Q Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def power_constraint(X_batch, B, Q):
    power = torch.mean(torch.sum(X_batch ** 2, dim=1))
    if power > B:
        X_batch = X_batch * torch.sqrt(B / power)
    return X_batch

def main():
    num_bits = 1000
    B_values = [1, 2, 3]
    Q_values = [2, 4, 6]
    alpha_values = [0.5, 1.0, 1.5]

    BER_results = evaluate(B_values, Q_values, alpha_values)
    plot(BER_results, B_values, Q_values, alpha_values)

if __name__ == '__main__':
    main()
