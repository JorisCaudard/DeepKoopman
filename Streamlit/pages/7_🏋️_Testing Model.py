import streamlit as st
import matplotlib.pyplot as plt

###Importing relative files didn't work, I'll just copy paste the code of useful functions here (Ignore frome line 4 to line 247)
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryDataset(Dataset):
    """Création des datasets de trajectoires à partir des fichiers raw.

    Attributes:
        trajectoryDataset (dataframe): Dataframe à une colonne contenant chaque trajectoire construite.

    Methods:
        __init__: Initialise les paramètres du Dataset.
        __len__: Calcule la longueur du Dataset.
        __getitem___: Sort la i-eme trajectoire du Dataset.

    Example:
        Création du dataset d'entraînement.
        trajectoryDataset = TrajectoryDataset('Koopman Final Files/Data/DiscreteSpectrumExample_train.csv')
    """
    def __init__(self, filePath):
        """Initialise les paramètres du Dataset.

        Args:
            filePath (str): Chemin du fichier raw de train/test.
        """
        
        #Load the dataset
        fullDataset = pd.read_csv(filePath, header= None)

        #Create the target from the dataset (i.e the full trajectory of 50 steps)
        trajectoryList = []

        for i in range(0, len(fullDataset), 51):
            rows = fullDataset.iloc[i:i+50]
            trajectory = [torch.tensor(tuple(row), dtype= torch.float64) for index, row in rows.iterrows()]
            trajectoryList.append(trajectory)
            

        self.trajectoryDataset = pd.DataFrame({'Trajectories': trajectoryList})

    def __len__(self):
        """Calcule la longueur du Dataset.

        Args:
            None.

        Returns:
            int: Nombre de trajectoires du Dataset.
        """
        return self.trajectoryDataset.shape[0]
    
    def __getitem__(self, index):
        """Sort la i-eme trajectoire du Dataset.

        Args:
            index (int): Index de la trajectoire à extraire.

        Returns:
            list: Trajectoire du Dataset à l'index i.
        """
        return self.trajectoryDataset.loc[index, 'Trajectories']

def getDataLoader(dataset, batchSize=128):
    """Sort la i-eme trajectoire du Dataset.

    Args:
        dataset (Dataset): Dataset à transformer en Dataloader.
        batchsize (int, optional): Batch Size du Dataloader. Par défaut, 128.

    Returns:
        Dataloader: Dataloader correspondant au Dataset
    """
    return DataLoader(dataset, batch_size= batchSize)

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

  st.write(f'Epoch {epoch}, loss = {epLoss/size}')

  return koopmanModel

###Maine code is here

st.set_page_config(
    page_title="Testing a Model",
    page_icon="🏋️",
    layout= 'wide'
)

st.title("Tester le modèle")

st.markdown("On implémente ici les fonctions d'entraînement dans modèle.")

modelParams = {
    'numShifts':50,
    'inputSize':2,
    'hiddenLayer':2,
    'hiddenSize':50,
    'latentSize':3

}

useCustomParams = st.toggle('Utiliser des paramètres custom ?')

if useCustomParams:
    modelParams["numShifts"] = st.slider("numShifts", 1, 50, 30)
    modelParams["hiddenLayer"] = st.slider("hiddenLayer", 0, 10, 2)
    modelParams["hiddenSize"] = st.slider("hiddenSize", 10, 100, 50, 5)
    modelParams["latentSize"] = st.slider("latentSize", 0, 10, 3)

    if st.button("Train model", type='primary'):
        with st.spinner("Data Loading ..."):
            testDataLoader = getDataLoader(TrajectoryDataset('Data/DiscreteSpectrumExample_train.csv'))

        st.success('Data Loaded !')

        with st.spinner("Model and Loss Builiding ..."):
            testKoopmanModel = SimpleKoopmanNeuralNetwork(modelParams)
            testKoopmanModel = testKoopmanModel.to(torch.float64)

            testLoss = LossFunction(modelParams)

        st.success('Model Built !')

        with st.spinner("Training model ..."):
            epochs = 100

            for t in range(epochs):
                pytorchTraining(testKoopmanModel, testLoss, torch.optim.Adam(testKoopmanModel.parameters()), t, testDataLoader)

        st.success("Model trained !")

        torch.save(testKoopmanModel.state_dict(), 'Model/customTrainedModel.pt')

if useCustomParams:
    testModel = SimpleKoopmanNeuralNetwork(modelParams)

    testModel.load_state_dict(torch.load('Model/customTrainedModel.pt'))
    testModel = testModel.to(torch.float64)

else:
    testModel = SimpleKoopmanNeuralNetwork(modelParams)

    testModel.load_state_dict(torch.load('Model/trainedModel.pt'))
    testModel = testModel.to(torch.float64)

st.markdown("## Input un état initial")

col1, col2 = st.columns(2)

with col1:
     x0 = st.number_input("Input X initial coordinate", -1., 1., 0.)

with col2:
     x1 = st.number_input("Input Y initial coordinate", -1., 1., 0.)

initialState = torch.tensor([x0, x1], dtype= torch.float64)

if st.button("Compute trajectory"):

    trajectoryPrediction = testModel(initialState)

    trajectoryX = [tensor[0].item() for tensor in trajectoryPrediction]
    trajectoryY = [tensor[1].item() for tensor in trajectoryPrediction]

    fig, ax = plt.subplots()

    ax.plot(trajectoryX, trajectoryY)
    ax.scatter(trajectoryX, trajectoryY, color= 'red')

    from scipy.integrate import odeint

    def model(y,t):
        x1, x2 = y
        dy_dt = [-0.05 * x1, -1 * (x2 - x1**2)]
        return dy_dt
    
    y0 = [x0, x1]
    import numpy as np
    t = np.arange(0,1,0.02)

    sol = odeint(model, y0, t)

    ax.plot(sol[:, 0], sol[:, 1], color='green')

    st.pyplot(fig)

    st.markdown("Comme on peut le voir, sur une trajectoire, le résultat semble assez approximatif ; cela est sûrement du à la simplification de la structure du réseau et la simplification de la fonction de perte dans le cadre de notre étude.")