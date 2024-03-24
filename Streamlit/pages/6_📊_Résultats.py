import streamlit as st

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