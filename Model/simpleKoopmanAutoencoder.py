import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dataloader

class SimpleKoopmanNeuralNetwork(nn.Module):
    """Classe représentant un réseau de neurones simple de Koopman.
    Ce réseau de neurones prend un état en entrée et prédit la trajectoire correspondante.

    Attributes:
        params (dict): Paramètres relatifs à la structure du réseau.
        Encoder (nn.Sequential): Encoder du réseau. 
        Decoder (nn.Sequential): Decoder du réseau.
        K (nn.Linear): Matrice de l'opérateur de Koopman, identifiée à une couche nn.Linear sans biais.

    Methods:
        __init__: Initialise les paramètres du réseau de neurones.
        forward: Prédit une trajectoire à partir d'un état initial
    """

    def __init__(self, params):
        """Initialise le réseau de neurones avec la configuration spécifiée.

        Args:
            params (dict): Un dictionnaire contenant la configuration initiale du réseau de neurones.
                Ce dictionnaire doit inclure les paramètres inputSize, hiddenSize, hiddenLayer, latentSize.
        """
        super(SimpleKoopmanNeuralNetwork, self).__init__()
        
        self.params = params

        inputSize, hiddenSize, hiddenLayer, latentSize = self.params['inputSize'], self.params['hiddenSize'], self.params['hiddenLayer'], self.params['latentSize']

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

    def forward(self, initialInput):
        """Prédit une trajectoire de longueur 50 à partir d'un état initial.

        Args:
            intialInput (nn.Tensor): État initial, de dimension inputSize.

        Returns:
            trajectoryPrediction (list): Trajectoire prédite sur 50 steps
        """
        #Take as input a 2 dimension tensor (initial State)

        #get the first encoded state (y1)
        encodedInitialInput = self.Encoder(initialInput)

        #First element of the latent trajectory is encoded input
        latentTrajectoryPrediction = [encodedInitialInput]

        #Alongside the trajectory, we multiply by the Koopman operator
        for _ in range(49):
            latentTrajectoryPrediction.append(self.K(latentTrajectoryPrediction[-1]))

        #Decoding the trajectory
        trajectoryPrediction = []

        for latentState in latentTrajectoryPrediction:
            trajectoryPrediction.append(self.Decoder(latentState))
        
        #We output the trajectoryPrediction
        return trajectoryPrediction
    
    

class LossFunction(nn.Module):
    """Classe contenant la fonction de perte du réseau de Koopman.

    Args:
        numShifts (int): Longueur de la trajectoire sur laquelle la fonction de perte est évaluée.
    """
    def __init__(self, params):
        """Initialise la fonction de perte avec les paramètres donnés.

        Args:
            params (dict): Un dictionnaire contenant la configuration initiale de la fonction de perte.
                Ce dictionnaire doit inclure le paramètre numShifts.
        """
        super().__init__()

        self.numShifts = params['numShifts']

    def forward(self, targetTrajectory, predictedTrajectory):
        """Calcule la fonction de perte entre deux trajectoires.

        Args:
            targetTrajectory (list): Trajectoire réelle à approcher.
            predictedTrajectory (list): Trajdectoire prédite par un réseau de neurones de Koopman*

        Returns:
            lossPred (torch.Tensor): Perte calculée entre les deux trajectoires. Le torch.Tensor contient la valeur de la fonction de perte et le gradient associé. 
        """

        #We compute the Prediction loss
        lossPred = 0

        for m in range(self.numShifts):
            lossPred += F.mse_loss(targetTrajectory[m], predictedTrajectory[m])

        return lossPred
    


if __name__ == '__main__':

    #Initializing the parameters dictionary
    params = {}

    #Settings related to loss function
    params['numShifts'] = 50
    #Settings related to Network Architecture
    params['inputSize'] = 2
    params['hiddenSize'] = 30
    params['hiddenLayer'] = 2
    params['latentSize'] = 2


    testKoopmanModel = SimpleKoopmanNeuralNetwork(params)
    testKoopmanModel = testKoopmanModel.to(torch.float64)

    testDataset = dataloader.TrajectoryDataset('Data/DiscreteSpectrumExample_train.csv')

    testLoss = LossFunction(params)

    print(testLoss(testDataset[0], testKoopmanModel(testDataset[0][0])))

    print(testKoopmanModel(testDataset[0][0]))