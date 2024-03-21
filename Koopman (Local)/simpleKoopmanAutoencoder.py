import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleKoopmanNeuralNetwork(nn.Module):
    """Simple Koopman Neural Network model for an autoencoder.

    This class implements a simple Koopman Neural Network model for an autoencoder,
    consisting of an encoder, a decoder, and a Koopman operator.

    Parameters
    ----------
    inputSize : int
        The size of the input data.
    hiddenSize : int
        The size of the hidden layers in the encoder and decoder.
    hiddenLayer : int
        The number of hidden layers in the encoder and decoder.
    latentSize : int
        The size of the latent space representation.

    Attributes
    ----------
    Encoder : torch.nn.Sequential
        The encoder network.
    Decoder : torch.nn.Sequential
        The decoder network.
    K : torch.nn.Linear
        The Koopman operator.

    Methods
    -------
    forward(x)
        Forward pass of the autoencoder.
    multiForward(x, numShift)
        Performs multiple forward passes of the autoencoder.

    """

    def __init__(self, inputSize, hiddenSize, hiddenLayer, latentSize):
        super(SimpleKoopmanNeuralNetwork, self).__init__()
        
        #Input layer of Encoder
        encoderLayers = [nn.Linear(inputSize, hiddenSize), nn.ReLU()]

        ###Define Encoder Layer
        for _ in range(hiddenLayer):
            encoderLayers.append(nn.Linear(hiddenSize, hiddenSize))
            encoderLayers.append(nn.ReLU())

        #Output layer of Encoder
        encoderLayers.append(nn.Linear(hiddenSize, latentSize))

        #Creating the Encoder Network
        self.Encoder = nn.Sequential(*encoderLayers)

        #Input layer of Decoder
        decoderLayers = [nn.Linear(latentSize, hiddenSize), nn.ReLU()]

        ###Define Decoder Layer
        for _ in range(hiddenLayer):
            decoderLayers.append(nn.Linear(hiddenSize, hiddenSize))
            decoderLayers.append(nn.ReLU())

        #Output layer of Decoder
        decoderLayers.append(nn.Linear(hiddenSize, inputSize))

        #Creating the Decoder Network
        self.Decoder = nn.Sequential(*decoderLayers)

        #Simple Koopman Auto-Encoder (Without Auxiliary Network)
        self.K = nn.Linear(latentSize, latentSize, bias=False)

    def forward(self, x):
        """Forward pass of the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Encoded representation of the input.
        torch.Tensor
            Koopman-transformed representation of the encoded input.
        torch.Tensor
            Decoded output.
        torch.Tensor
            Encoded representation of the decoded output.

        """

        xEncoded = self.Decoder(self.Encoder(x))

        y = self.Encoder(x)
        yNext = self.K(y)
        xNext = self.Decoder(yNext)

        return y, yNext, xNext, xEncoded
    
    def multiForward(self, x, numShift):
        """Performs multiple forward passes of the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        numShift : int
            Number of forward passes.

        Returns
        -------
        list
            List of decoded outputs for each forward pass.

        """

        xNextList = []

        for _ in range(numShift):
            _, _, xNext, _ = self.forward(x)
            xNextList.append(xNext.clone())
            x = xNext

        return xNextList
    
    

class SimpleLossFunction(nn.Module):
    """Simple loss function for training the autoencoder.

    This class implements a simple loss function for training the autoencoder.

    Parameters
    ----------
    alpha1 : float
        Weight for the reconstruction loss.
    alpha2 : float
        Weight for the prediction loss.
    alpha3 : float
        Weight for the weight regularization loss.

    Methods
    -------
    lossRecon(x, xEncoded)
        Computes the reconstruction loss.
    lossPred(x, xNextList)
        Computes the prediction loss.
    lossInf(x, xEncoded, xNext, xNextEncoded)
        Computes the infinity norm loss.
    lossWeight(model)
        Computes the weight regularization loss.
    forward(model, x, xEncoded, xNextList, xNext, xNextEncoded)
        Computes the total loss.

    """

    def __init__(self, alpha1, alpha2, alpha3):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

    def lossRecon(self, x, xEncoded):
        """Computes the reconstruction loss.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        xEncoded : torch.Tensor
            Encoded representation of the input.

        Returns
        -------
        torch.Tensor
            Reconstruction loss.

        """
        
        return F.mse_loss(x, xEncoded)
    
    def lossPred(self, x, xNextList):
        """Compute the prediction loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        xNextList : list of torch.Tensor
            List of predicted tensors for future time steps.

        Returns
        -------
        torch.Tensor
            Mean squared error loss between the input and predicted tensors.

        """

        return torch.mean(torch.stack([F.mse_loss(x, xNext) for xNext in xNextList]))

    def lossInf(self, x, xEncoded, xNext, xNextEncoded):
        """Compute the infinity norm loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        xEncoded : torch.Tensor
            Encoded tensor.
        xNext : torch.Tensor
            Tensor for the next time step.
        xNextEncoded : torch.Tensor
            Encoded tensor for the next time step.

        Returns
        -------
        torch.Tensor
            Infinity norm loss between the input and encoded tensors, and between the next time step tensors and their encoded tensors.

        """

        return torch.linalg.vector_norm(x - xEncoded, ord = np.inf) + torch.linalg.vector_norm(xNext - xNextEncoded, ord = np.inf)

    def lossWeight(self, model):
        """Calculate the regularization term on the weights of the neural network model.

        Parameters
        ----------
        model : nn.Module
            The neural network model.

        Returns
        -------
        torch.Tensor
            The computed weight norm loss.
            
        """
        lossWeight = 0

        for key, item in model.state_dict().items():
            parts = key.split(".")

            if parts[0] in ["Encoder", "Decoder"] and parts[-1] == "weight":

                lossWeight += torch.linalg.matrix_norm(item)

        return lossWeight

    def forward(self, model, x, xEncoded, xNextList, xNext, xNextEncoded):
        """Forward pass of the loss function.

        Parameters
        ----------
        model : nn.Module
            Neural network model.
        x : torch.Tensor
            Input tensor.
        xEncoded : torch.Tensor
            Encoded tensor.
        xNextList : list of torch.Tensor
            List of predicted tensors for future time steps.
        xNext : torch.Tensor
            Tensor for the next time step.
        xNextEncoded : torch.Tensor
            Encoded tensor for the next time step.

        Returns
        -------
        torch.Tensor
            Loss value computed based on the input and predicted tensors.

        """

        return self.alpha1 * (self.lossRecon(x, xEncoded) + self.lossPred(x, xNextList)) + self.alpha2 * self.lossInf(x, xEncoded, xNext, xNextEncoded) + self.alpha3 * self.lossWeight(model)