import torch.nn as nn
from functions import ReverseLayerF
import torch.nn.functional as F

class ViT(nn.Module):

    def __init__(self, latentVecSize) -> None:
        super(ViT, self).__init__()
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latentVecSize, nhead=4, dim_feedforward=512, activation="relu",
                                       batch_first=True),
            num_layers=2
        )

    def forward(self, x):
        x = self.encoder1(x)
        return x


class BANNModel(nn.Module):

    def __init__(self):
        super(BANNModel, self).__init__()

        self.feature = nn.Sequential()

        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=3, padding=1))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))


        self.gru1 = nn.GRU(input_size=400, hidden_size=400, num_layers=2, batch_first=True)
        self.mlp11 = nn.Linear(400, 100)
        self.BN11 = nn.BatchNorm1d(100)
        self.relu11 = nn.ReLU(True)
        self.mlp12 = nn.Linear(100, 2)


        self.transformerEncoder = ViT(latentVecSize=400)
        self.mlp21 = nn.Linear(400, 100)
        self.BN21 = nn.BatchNorm1d(100)
        self.relu21 = nn.ReLU(True)
        self.mlp22 = nn.Linear(100, 4)


    def forward(self, input_data, lamuda):

        input_data = input_data.view(-1, 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 8, 50*8)
        reverse_feature = ReverseLayerF.apply(feature, lamuda)

        # User classifier
        x1, _ = self.gru1(feature)
        x1 = x1[:, -1, :] #output the last time step data
        x1 = self.mlp11(x1)
        x1 = self.BN11(x1)
        x1 = self.relu11(x1)
        x1 = self.mlp12(x1)
        class_output = F.log_softmax(x1, dim=1)

        # Behavior classifier
        x2 = self.transformerEncoder(reverse_feature)
        x2 = x2[:, 0] #output the first position data
        x2 = self.mlp21(x2)
        x2 = self.BN21(x2)
        x2 = self.relu21(x2)
        x2 = self.mlp22(x2)
        domain_output = F.log_softmax(x2, dim=1)

        return class_output, domain_output
