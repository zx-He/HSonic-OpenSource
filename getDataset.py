
import torch
from torch.utils.data import Subset

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
