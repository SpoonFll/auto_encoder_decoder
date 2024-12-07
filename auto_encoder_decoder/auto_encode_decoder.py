import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Parameters
samples = 1000
alpha = 0.7


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def power_constraint(X_batch,B):
    power = torch.mean(torch.sum(X_batch ** 2, dim=1))
    if power > B:
        X_batch = X_batch * torch.sqrt(B / power)
    return X_batch

class Auto_Encoder(nn.Module):
    q=0
    B=0
    Q=0
    def __init__(self,q,B,Q):
        super(Auto_Encoder, self).__init__()
        self.q=q
        self.B=B
        self.Q=Q

        #Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.q, self.q),
            nn.ReLU()
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.q, 1000),
            nn.Sigmoid()
        )
    def forward(self,input_bits):
        #batch_size = input_bits.size(0)
        input_bits = torch.reshape(input_bits,(int(1000/self.q),self.q))
        x = self.encoder(input_bits)
        #print(x)
        #x = power_constraint(x,B,Q)
        #indices = torch.randint(0, M, (batch_size,)).to(device)  # Shape: (batch_size,)
        S_i = torch.nn.functional.one_hot(x.long(),2).float().to(device)  # Shape: (batch_size, M)
        #print(S_i)
        #I = x*S_i
        #I = torch.sum(x * S_i, dim=1, keepdim=True)
        print(x.size())
        print(S_i.size())
        N_i = torch.randn(batch_size, 1).to(device)  # Shape: (batch_size, 1)
        I = I + N_i

        output_bits= self.decoder(I)
        Z_i=(torch.norm(output_bits)>=alpha).float().unsqueeze(1)
        return (output_bits,Z_i)


def train(epochs):
    B_values = [1, 2, 3]
    Q_values = [2, 4, 6]
    q_values=[2,4,6]
    model = Auto_Encoder(2,4,1)
    optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)
    for epoch in range(epochs):
        for i in range(3):
            sig = torch.tensor(np.random.randint(2,size=samples)).to(torch.float32)
            reconstructed = model(sig)
            print(reconstructed)




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


def main():
    num_bits = 1000
    B_values = [1, 2, 3]
    Q_values = [2, 4, 6]
    q_values=[2,4,6]
    alpha_values = [0.5, 1.0, 1.5]

    #BER_results = evaluate(B_values, Q_values, alpha_values)
    #plot(BER_results, B_values, Q_values, alpha_values)

    train(10)
if __name__ == '__main__':
    main()
