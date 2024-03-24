import simpleKoopmanAutoencoder as kp
import dataloader as dl
import torch


def pytorchTraining(koopmanModel, koopmanLoss, optimizer, epoch, dataloader):
  """Fonction d'entraînement du modèle.

  Args:
    koopmanModel (SimpleKoopmanNeuralNetwork): Une instance d'un modèle d'auto-encoder de Koopman.
    koopmanLoss (LossFunction): Fonction de perte à optimiser.
    optimizer (torch.optim): Optimizer utilisé pour l'optimisation.
    epoch (int): Numéro de l'epoch.
    dataloader (Dataloader): Dataloader des données d'entraînement.

  Returns:
    koopmanModel (SimpleKoopmanNeuralNetwork): Trained model. Also saved in a "trainedModel.pt" file.
  
  """
  
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

  #Settings related to loss function
  params['numShifts'] = 50

  #Settings related to Network Architecture
  params['inputSize'] = 2
  params['hiddenSize'] = 50
  params['hiddenLayer'] = 2
  params['latentSize'] = 3


  testKoopmanModel = kp.SimpleKoopmanNeuralNetwork(params)
  testKoopmanModel = testKoopmanModel.to(torch.float64)

  testDataset = dl.TrajectoryDataset('Data/DiscreteSpectrumExample_train.csv')

  testDataloader = dl.getDataLoader(testDataset)


  testLoss = kp.LossFunction(params)

  epochs = 100

  for t in range(epochs):
    pytorchTraining(testKoopmanModel, testLoss, torch.optim.Adam(testKoopmanModel.parameters()), t, testDataloader)
  
  torch.save(testKoopmanModel.state_dict(), 'Model/trainedModel.pt')
