import torch
import simpleKoopmanAutoencoder as kp
import dataloader as dl
import matplotlib.pyplot as plt
import numpy as np

params = {}

#Settings related to dataset
params['lenTime'] = 51
params['deltaT'] = 0.02

#Settings related to loss function
params['numShifts'] = 50

#Settings related to Network Architecture
params['inputSize'] = 2
params['hiddenSize'] = 50
params['hiddenLayer'] = 2
params['latentSize'] = 3

valDataset = dl.TrajectoryDataset('Koopman Final Files/Data/DiscreteSpectrumExample_train.csv')

testModel = kp.SimpleKoopmanNeuralNetwork(params)

testModel.load_state_dict(torch.load('Koopman Final Files/Model+Loss/trainedModel.pt'))
testModel = testModel.to(torch.float64)

print(valDataset[0])

for i in range(1):
    print(i)
    trajectoryX = [tensor[0].item() for tensor in valDataset[i]]
    trajectoryY = [tensor[1].item() for tensor in valDataset[i]]

    plt.plot(trajectoryX, trajectoryY, color= 'green')

   
    trajectoryPrediction = testModel(valDataset[i][0])
    print(trajectoryPrediction)

    trajectoryX = [tensor[0].item() for tensor in trajectoryPrediction]
    trajectoryY = [tensor[1].item() for tensor in trajectoryPrediction]

    plt.plot(trajectoryX, trajectoryY)
    plt.scatter(trajectoryX, trajectoryY, color= 'red')

plt.show()
