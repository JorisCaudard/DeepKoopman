import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TrajectoryDataset(Dataset):
    def __init__(self, filePath):
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
        return self.trajectoryDataset.shape[0]
    
    def __getitem__(self, index):
        return self.trajectoryDataset.loc[index, 'Trajectories']

def getDataLoader(dataset, batchSize=128):
    return DataLoader(dataset, batch_size= batchSize)



if __name__ == '__main__':

    testDataset = TrajectoryDataset('Koopman Final Files/Data/DiscreteSpectrumExample_train.csv') 
    print(testDataset[3])
    testDataloader = getDataLoader(testDataset)
    print(testDataloader)
    print(len(testDataset))
