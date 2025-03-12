import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractorModality1(nn.Module):
    def __init__(self):
        super(FeatureExtractorModality1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        return x

class FeatureExtractorModality2(nn.Module):
    def __init__(self):
        super(FeatureExtractorModality2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        return x

class AttentionFusionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionFusionLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)
        attention_weights = F.softmax(self.fc2(F.relu(self.fc1(combined))), dim=1)
        fused_features = attention_weights * combined
        return fused_features

class Classifier(nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.feature_extractor1 = FeatureExtractorModality1()
        self.feature_extractor2 = FeatureExtractorModality2()
        self.attention_fusion = AttentionFusionLayer(input_dim=240, hidden_dim=120)
        self.classifier = Classifier()

    def forward(self, x1, x2):
        features1 = self.feature_extractor1(x1)
        features2 = self.feature_extractor2(x2)
        fused_features = self.attention_fusion(features1, features2)
        output = self.classifier(fused_features)
        return output