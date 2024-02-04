import torch
import torch.nn as nn
import torch.nn.functional as F


class KoopmanNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, auxiliary_size):
        super(KoopmanNeuralNetwork, self).__init__()
        ###Define Encoder
        self.Encoder = nn.Sequential(
            nn.Linear(in_features= input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

        ###Define Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

        ###Define Koopman Auxiliary Network
        self.Auxiliary = nn.Sequential(
            nn.Linear(latent_size, auxiliary_size),
            nn.ReLU(),
            nn.Linear(auxiliary_size, auxiliary_size),
            nn.ReLU(),
            nn.Linear(auxiliary_size, latent_size),
        )

    def forward(self, x_k):
        y_k = self.Encoder(x_k)
        y_k1 = self.Auxiliary(y_k)
        x_k1 = self.Decoder(y_k1)

        return x_k, y_k, y_k1, x_k1