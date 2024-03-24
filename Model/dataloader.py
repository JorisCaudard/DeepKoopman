import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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



if __name__ == '__main__':

    testDataset = TrajectoryDataset('Data/DiscreteSpectrumExample_train.csv') 
    print(testDataset[3])
    testDataloader = getDataLoader(testDataset)
    print(testDataloader)
    print(len(testDataset))
