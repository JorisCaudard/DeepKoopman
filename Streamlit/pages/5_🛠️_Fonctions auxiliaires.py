import streamlit as st

st.set_page_config(
    page_title="Training/testing",
    page_icon="🛠️",
    layout= 'wide'
)

st.title("Fonctions utilitaires")

tab1, tab2 = st.tabs(["Data Loading", "Train function"])

with tab1:
    st.header("Data Loading")

    st.markdown("Les fichiers de données raw contiennent des trajectoires possibles (calculées en résolavnt des équations différentielles). Il est nécessaire pour l'entraîenement du modèle de les transformer en Dataset (classe du module pytorch), afin de pouvoir entraîner les modèles. On crée aussi un Dataloader permettant d'itérer sur le Dataset pour l'entraînement du modèle.")

    st.subheader("Classe Dataset")
    st.code("""
class TrajectoryDataset(Dataset):
    \"""Création des datasets de trajectoires à partir des fichiers raw.

    Attributes:
        trajectoryDataset (dataframe): Dataframe à une colonne contenant chaque trajectoire construite.

    Methods:
        __init__: Initialise les paramètres du Dataset.
        __len__: Calcule la longueur du Dataset.
        __getitem___: Sort la i-eme trajectoire du Dataset.

    Example:
        Création du dataset d'entraînement.
        trajectoryDataset = TrajectoryDataset('Koopman Final Files/Data/DiscreteSpectrumExample_train.csv')
    \"""
    def __init__(self, filePath):
        \"""Initialise les paramètres du Dataset.

        Args:
            filePath (str): Chemin du fichier raw de train/test.
        \"""
        
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
        \"""Calcule la longueur du Dataset.

        Args:
            None.

        Returns:
            int: Nombre de trajectoires du Dataset.
        \"""
        return self.trajectoryDataset.shape[0]
    
    def __getitem__(self, index):
        \"""Sort la i-eme trajectoire du Dataset.

        Args:
            index (int): Index de la trajectoire à extraire.

        Returns:
            list: Trajectoire du Dataset à l'index i.
        \"""
        return self.trajectoryDataset.loc[index, 'Trajectories']
        """
    )

    st.subheader("Fonction Dataloader")

    st.code("""
    def getDataLoader(dataset, batchSize=128):
        \"""Sort la i-eme trajectoire du Dataset.

        Args:
            dataset (Dataset): Dataset à transformer en Dataloader.
            batchsize (int, optional): Batch Size du Dataloader. Par défaut, 128.

        Returns:
            Dataloader: Dataloader correspondant au Dataset
        \"""
        return DataLoader(dataset, batch_size= batchSize)
            """)
    
    with tab2:
        st.header("Train loop function")

        st.markdown("Voici l'implémentation de la fonction d'entraînement utilisée ici :")

        st.code("""def pytorchTraining(koopmanModel, koopmanLoss, optimizer, epoch, dataloader):
  \"""Fonction d'entraînement du modèle.

  Args:
    koopmanModel (SimpleKoopmanNeuralNetwork): Une instance d'un modèle d'auto-encoder de Koopman.
    koopmanLoss (LossFunction): Fonction de perte à optimiser.
    optimizer (torch.optim): Optimizer utilisé pour l'optimisation.
    epoch (int): Numéro de l'epoch.
    dataloader (Dataloader): Dataloader des données d'entraînement.

  Returns:
    koopmanModel (SimpleKoopmanNeuralNetwork): Trained model. Also saved in a "trainedModel.pt" file.
  
  \"""
  
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
        

""")
        
        st.markdown("Il suffit alors de boucler sur le nombre d'epochs d'entraînement pour construire un modèle optimisé.")