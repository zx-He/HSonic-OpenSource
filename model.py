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


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=3, padding=1),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.feature(x).view(-1, 8, 50*8)

class UserClassifier(nn.Module):
    def __init__(self):
        super(UserClassifier, self).__init__()
        self.gru1 = nn.GRU(input_size=400, hidden_size=400, num_layers=2, batch_first=True)
        self.mlp11 = nn.Linear(400, 100)
        self.BN11 = nn.BatchNorm1d(100)
        self.relu11 = nn.ReLU(True)
        self.mlp12 = nn.Linear(100, 2)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = x[:, -1, :] #output the last time step data
        x = self.mlp11(x)
        x = self.BN11(x)
        x = self.relu11(x)
        x = self.mlp12(x)
        return F.log_softmax(x, dim=1)

class BehaviorClassifier(nn.Module):
    def __init__(self):
        super(BehaviorClassifier, self).__init__()
        self.transformerEncoder = ViT(latentVecSize=400)
        self.mlp21 = nn.Linear(400, 100)
        self.BN21 = nn.BatchNorm1d(100)
        self.relu21 = nn.ReLU(True)
        self.mlp22 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.transformerEncoder(x)
        x = x[:, 0] #output the first position data
        x = self.mlp21(x)
        x = self.BN21(x)
        x = self.relu21(x)
        x = self.mlp22(x)
        return F.log_softmax(x, dim=1)

class BANNModel(nn.Module):
    def __init__(self):
        super(BANNModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.user_classifier = UserClassifier()
        self.behavior_classifier = BehaviorClassifier()

    def forward(self, input_data, lamuda):
        input_data = input_data.view(-1, 3, 32, 32)
        feature = self.feature_extractor(input_data)

        class_output = self.user_classifier(feature)

        reverse_feature = ReverseLayerF.apply(feature, lamuda)
        domain_output = self.behavior_classifier(reverse_feature)

        return class_output, domain_output


class BANNModel_transfer(nn.Module):
    def __init__(self):
        super(BANNModel_transfer, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.user_classifier = UserClassifier()

    def forward(self, input_data):
        input_data = input_data.view(-1, 3, 32, 32)
        feature = self.feature_extractor(input_data)
        class_output = self.user_classifier(feature)

        return class_output