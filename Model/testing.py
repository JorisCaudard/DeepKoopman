import torch
import simpleKoopmanAutoencoder as kp
import dataloader as dl
import matplotlib.pyplot as plt
import numpy as np

params = {}

#Settings related to loss function
params['numShifts'] = 50

#Settings related to Network Architecture
params['inputSize'] = 2
params['hiddenSize'] = 50
params['hiddenLayer'] = 2
params['latentSize'] = 3

valDataset = dl.TrajectoryDataset('Data/DiscreteSpectrumExample_val_x.csv')

testModel = kp.SimpleKoopmanNeuralNetwork(params)

testModel.load_state_dict(torch.load('Model/trainedModel.pt'))
testModel = testModel.to(torch.float64)


for i in range(len(valDataset)):
    trajectoryX = [tensor[0].item() for tensor in valDataset[i]]
    trajectoryY = [tensor[1].item() for tensor in valDataset[i]]

    plt.plot(trajectoryX, trajectoryY, color= 'green')

   
    trajectoryPrediction = testModel(valDataset[i][0])

    trajectoryX = [tensor[0].item() for tensor in trajectoryPrediction]
    trajectoryY = [tensor[1].item() for tensor in trajectoryPrediction]

    plt.plot(trajectoryX, trajectoryY)
    plt.scatter(trajectoryX, trajectoryY, color= 'red')

plt.show()