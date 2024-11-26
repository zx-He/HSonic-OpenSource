# -*- coding:utf-8 -*-
"""
作者：${何志想}
日期：2023年06月04日
"""

import os
import torch
from torch.utils.data import Dataset, Subset
import math
import numpy as np


def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = np.array([list(map(float, line.strip().split(','))) for line in lines])
    return data

class getDatasetwithoutDomain(Dataset):
    def __init__(self, root_dir, train=True, UserList=[1,2], dataTypeList=["chair_10"]):
        self.root_dir = root_dir
        self.samples = []
        self.labels = []
        self.train = train

        for i in UserList:
            User_path = os.path.join(root_dir, str(i))
            for dataType in dataTypeList:
                folder_path = os.path.join(User_path, dataType)
                for j in range(1, 101):
                    sample_path = os.path.join(folder_path, str(j) + ".txt")
                    data = read_data(sample_path)
                    data = data.flatten()
                    self.samples.append(data)
                    self.labels.append(i)
        self.samples = torch.tensor(self.samples).float()
        self.labels = torch.tensor(self.labels).float()

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return torch.tensor(sample), torch.tensor(label)

    def __len__(self):
        return len(self.samples)

class getDatasetwithDoamin(Dataset):
    def __init__(self, root_dir, train=True, UserList=[], dataType = ['chair_10'], domainLable=0):
        self.root_dir = root_dir
        self.samples = []
        self.labels = []
        self.domainLabels = []
        self.train = train
        self.userList = UserList
        self.dataType = dataType

        for i in range(len(UserList)):
            UserPath = os.path.join(root_dir, str(UserList[i]))
            for type in dataType:
                dataType_path = os.path.join(UserPath, type)
                for j in range(1, 101):
                    file_path = os.path.join(dataType_path, str(j) + ".txt")
                    data = read_data(file_path)
                    data = data.flatten()
                    self.samples.append(data)
                    self.labels.append(i)
                    self.domainLabels.append(domainLable)

        self.samples = torch.tensor(self.samples).float()
        self.labels = torch.tensor(self.labels).long()
        self.domainLabels = torch.tensor(self.domainLabels).long()
        self.samples = torch.unsqueeze(self.samples, 1)


    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        domainLabel = self.domainLabels[index]
        return sample, label, domainLabel

    def __len__(self):
        return len(self.samples)

class getIllegalDataset(Dataset):
    def __init__(self, root_dir, train=True, UserList=[1,2], legalUser = 0, dataTypeList=["chair_10"]):
        self.root_dir = root_dir
        self.samples = []
        self.labels = []
        self.train = train
        self.legalUser = legalUser


        for i in UserList:
            if i != legalUser:
                User_path = os.path.join(root_dir, str(i))
                for dataType in dataTypeList:
                    folder_path = os.path.join(User_path, dataType)
                    for j in range(1, 101):
                        sample_path = os.path.join(folder_path, str(j) + ".txt")
                        data = read_data(sample_path)
                        data = data.flatten()
                        self.samples.append(data)
                        self.labels.append(i)
        self.samples = torch.tensor(self.samples).float()
        self.labels = torch.tensor(self.labels).float()

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return torch.tensor(sample), torch.tensor(label)

    def __len__(self):
        return len(self.samples)


class GetSubset(Subset):
    def __init__(self, dataset, indices, train=True, transform=None):
        super().__init__(dataset, indices)
        self.train = train
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

