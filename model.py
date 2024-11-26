import torch.nn as nn
from functions import ReverseLayerF
import torch.nn.functional as F

class ViT(nn.Module):

    def __init__(self, latentVecSize) -> None:
        super(ViT, self).__init__()
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latentVecSize, nhead=5, dim_feedforward=512, activation="relu",
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
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(3),
            nn.ReLU(True),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.Conv1d(32, 10, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
        )

    def forward(self, x):
        return self.feature(x).view(-1, 13, 10)

class UserClassifier(nn.Module):
    def __init__(self):
        super(UserClassifier, self).__init__()
        self.gru1 = nn.GRU(input_size=10, hidden_size=10, num_layers=2, batch_first=True)
        self.mlp11 = nn.Linear(10, 100)
        self.BN11 = nn.BatchNorm1d(100)
        self.relu11 = nn.ReLU(True)
        self.mlp12 = nn.Linear(100, 60)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = x[:, -1, :]
        x = self.mlp11(x)
        x = self.BN11(x)
        x = self.relu11(x)
        x = self.mlp12(x)
        return F.log_softmax(x, dim=1)


class BehaviorClassifier(nn.Module):
    def __init__(self):
        super(BehaviorClassifier, self).__init__()
        self.transformerEncoder = ViT(latentVecSize=10)
        self.mlp21 = nn.Linear(10, 100)
        self.BN21 = nn.BatchNorm1d(100)
        self.relu21 = nn.ReLU(True)
        self.mlp22 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.transformerEncoder(x)
        x = x[:, 0]
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
        input_data = input_data.view(-1, 1, 78)
        feature = self.feature_extractor(input_data)
        reverse_feature = ReverseLayerF.apply(feature, lamuda)

        class_output = self.user_classifier(feature)
        domain_output = self.behavior_classifier(reverse_feature)

        return class_output, domain_output


