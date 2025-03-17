import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Dynamically calculated later

        # Decoder for reconstruction
        self.fc2 = nn.Linear(120, 16 * 53 * 53)
        self.deconv1 = nn.ConvTranspose2d(16, 6, kernel_size=5, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(6, 3, kernel_size=5, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: (B, 6, 110, 110)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (B, 16, 53, 53)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        features = F.relu(self.fc1(x))

        # Reconstruction
        x = F.relu(self.fc2(features))
        x = x.view(-1, 16, 53, 53)
        x = F.relu(self.deconv1(x))  # Output: (B, 6, 109, 109)
        x = F.relu(self.deconv2(x))  # Output: (B, 3, 221, 221)
        reconstruction = torch.sigmoid(self.deconv3(x))  # Output: (B, 3, 224, 224)
        return features, reconstruction

class AttentionFusionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusionLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, input_dim)
        self.fc2 = nn.Linear(input_dim, 2)  # Two weights, one for each modality

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)  # Shape: (B, 240)
        attention_scores = self.fc2(F.relu(self.fc1(combined)))  # Shape: (B, 2)
        attention_weights = F.softmax(attention_scores, dim=1)  # Sum to 1

        # Apply attention weights to both modalities
        fused_features = (
            attention_weights[:, 0].unsqueeze(1) * x1
            + attention_weights[:, 1].unsqueeze(1) * x2
        )
        return fused_features

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 40)  # ModelNet40 has 40 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.feature_extractor1 = FeatureExtractor()
        self.feature_extractor2 = FeatureExtractor()
        self.attention_fusion = AttentionFusionLayer(input_dim=120)
        self.classifier = Classifier()

    def forward(self, x1, x2):
        features1, _ = self.feature_extractor1(x1)
        features2, _ = self.feature_extractor2(x2)
        fused_features = self.attention_fusion(features1, features2)
        output = self.classifier(fused_features)
        return output