import simpleKoopmanAutoencoder as kp
import dataloader as dl
import torch


def pytorchTraining(koopmanModel, koopmanLoss, optimizer, epoch, dataloader):
  
  koopmanModel.train()
  epLoss = 0
  size = len(dataloader)

  for target in dataloader:

    optimizer.zero_grad()

    initialState, trueTrajectory = target[0], target  

    predictedTrajectory = koopmanModel(initialState)

    loss = koopmanLoss(trueTrajectory, predictedTrajectory)

    epLoss += loss.item()

    loss.backward()
    optimizer.step()

  print(f'Epoch {epoch}, loss = {epLoss/size}')

  return koopmanModel


if __name__ == '__main__':
#Initializing the parameters dictionary
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


  testKoopmanModel = kp.SimpleKoopmanNeuralNetwork(params)
  testKoopmanModel = testKoopmanModel.to(torch.float64)

  testDataset = dl.TrajectoryDataset('Koopman Final Files/Data/DiscreteSpectrumExample_train.csv')

  testDataloader = dl.getDataLoader(testDataset)


  testLoss = kp.LossFunction(params)

  epochs = 50

  for t in range(epochs):
    pytorchTraining(testKoopmanModel, testLoss, torch.optim.Adam(testKoopmanModel.parameters()), t, testDataloader)
  
  torch.save(testKoopmanModel.state_dict(), 'Koopman Final Files/Model+Loss/trainedModel.pt')
