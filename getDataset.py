
import os
import torch
from torch.utils.data import Dataset, Subset
import numpy as np

class getOriginDataset(Dataset):
    def __init__(self, root_dir, train=True, UserList=[], dataType = ['chair_10']):
        self.root_dir = root_dir
        self.samples = []
        self.labels = []
        self.train = train
        self.userList = UserList
        self.dataType = dataType

        for i in UserList:
            UserPath = os.path.join(root_dir, str(i))
            for type in dataType:
                dataType_path = os.path.join(UserPath, type)
                for j in range(1, 101):
                    file_path = os.path.join(dataType_path, str(j) + ".txt")
                    data = np.loadtxt(file_path)
                    self.samples.append(data)
                    self.labels.append(i)

        self.samples = torch.tensor(self.samples).float()
        print("self.samples.shape:", self.samples.shape)
        self.labels = torch.tensor(self.labels).float()
        print("self.labels.shape:", self.labels.shape)
        self.samples = torch.unsqueeze(self.samples, 1)


    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return sample, label

    def __len__(self):
        return len(self.samples)


class GetSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
        self.samples = dataset.samples[indices]
        self.labels = dataset.labels[indices]

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return torch.tensor(sample), torch.tensor(label)

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    root_dir = '/root/project/Paper3_LeakPass/mainDataset'
    main_User = getOriginDataset(root_dir, startUser=1, endUser=46)