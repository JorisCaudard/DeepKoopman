import streamlit as st
import torch
import torch.nn as nn
import numpy as np

st.set_page_config(
    page_title="Results",
    page_icon="📊",
    layout= 'wide'
)

st.title("Résultats obtenus")

st.markdown("Après entraînement du modèle, on évalue les trajectoires obtenus sur un jeu de validation. Nous utiliserons le jeu de validation fourni par les rédacteurs de l'article Nature (Bethany Lusch).")

st.markdown("Pour ce faire, nous utiliserons un fichier python externe, ainsi que les fonctions Dataloader précédemennt utilisées pour l'entraînement du modèle.")

with st.expander("Voir le code utilisé"):
    st.code("""
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

valDataset = dl.TrajectoryDataset('Koopman Final Files/Data/DiscreteSpectrumExample_val_x.csv')

testModel = kp.SimpleKoopmanNeuralNetwork(params)

testModel.load_state_dict(torch.load('Koopman Final Files/Model+Loss/trainedModel.pt'))
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

""")
    
st.markdown("Voici les résultats obtenus sur le jeu de validation complet :")

st.image("Files/Results.png", caption="Trajectoire reconstruite par le réseau")

st.markdown("Comme on peut le voir sur cette image superposant les trajectoires cibles en vert et les trajectoires reconstruites en couleur, le modèle semble avoir appris la logique globale du système dynamique, ce qui justifie la faisabilité de ce modèle. En revanche, la généralisation reste brouillonne ; ceci est sûrement dû au modèle plus simple utilisé, mais également peut être ngendré par la fonction de perte simplifiée.")

st.subheader("Valeurs propres")

st.markdown("On peut également regarder les valeurs propres de la matrice de l'opérateur de Koopman. En théorie, on devrait retrouver les valeurs de $\lambda$, $\mu$ et $\lambda ²$ utilisés dans la construction du dataset (ici, $\lambda = -1$ et $\mu = -0.05$).")

###Couldn't import the Network as a module, so I'll just copy paste it instead. Ignore from here on out to line 165
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

###Resume reading code here
    
col1, col2 = st.columns(2)

with col1:
    st.markdown("Voici la matrice de l'opérateur apprise par le réseau :")

    modelParams = {
        'numShifts':50,
        'inputSize':2,
        'hiddenLayer':2,
        'hiddenSize':50,
        'latentSize':3

    }

    testModel = SimpleKoopmanNeuralNetwork(modelParams)

    testModel.load_state_dict(torch.load('Model/trainedModel.pt'))
    testModel = testModel.to(torch.float64)



    st.write(testModel.K.weight.detach().numpy())

with col2:
    st.markdown("Et voici ses valeurs propres :")

    st.write(np.linalg.eigvals(testModel.K.weight.detach().numpy()))

st.markdown("Dans notre cas, on sait que la forme générale de la matrice de l'opérateur K doit être :")

matriceLatex = r"""
\begin{pmatrix}
\mu & 0 & 0 \\
0 & \lambda & -\lambda \\
0 & 0 & 2\mu
\end{pmatrix}
"""

st.latex(matriceLatex)

st.markdown("Qui a pour valeurs propres $\lambda$, $\mu$. Dans notre cas, le réseau a du mal à identifier ces valeurs propres. L'utilisation du réseau auxiliaire proposé dans l'article peut permettre de palier ce problème.")