import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dataloader

class SimpleKoopmanNeuralNetwork(nn.Module):
    def __init__(self, params):
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
    def __init__(self, params):
        super().__init__()

        self.numShifts = params['numShifts']

    def forward(self, targetTrajectory, predictedTrajectory):

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

    testDataset = dataloader.TrajectoryDataset('Koopman (Local)/data/DiscreteSpectrumExample_train1_x.csv')

    testLoss = LossFunction(params)

    print(testLoss(testDataset[0], testKoopmanModel(testDataset[0][0])))

    print(testKoopmanModel(testDataset[0][0]))