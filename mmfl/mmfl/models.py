import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLayers(nn.Module):
    def __init__(self, input_size):
        super(CNNLayers, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),

            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),
        )

        # Decoder for reconstruction
        self.fc1 = nn.Linear(32 * 28 * 28, 32 * 28 * 28)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, input_size, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def reconstruct(self, features):
        x = F.relu(self.fc1(features))
        x = x.view(-1, 32, 28, 28)
        x = F.relu(self.deconv1(x))
        reconstruction = torch.sigmoid(self.deconv2(x))
        return reconstruction

class AttentionFusionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusionLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, input_dim)
        self.fc2 = nn.Linear(input_dim, 2)  # Two weights, one for each modality

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)  # Shape: (B, 32 * 28 * 28 * 2)
        attention_scores = self.fc2(F.relu(self.fc1(combined)))  # Shape: (B, 2)
        attention_weights = F.softmax(attention_scores, dim=1)  # Sum to 1

        # Apply attention weights to both modalities
        fused_features = (
            attention_weights[:, 0].unsqueeze(1) * x1
            + attention_weights[:, 1].unsqueeze(1) * x2
        )
        return fused_features

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CombinedModel(nn.Module):
    def __init__(self, input_size=3, num_classes=40):  # Default values for input_size and num_classes
        super(CombinedModel, self).__init__()
        self.cnn_layers1 = CNNLayers(input_size)
        self.cnn_layers2 = CNNLayers(input_size)
        self.attention_fusion = AttentionFusionLayer(input_dim=32 * 28 * 28)
        self.classifier = Classifier(32 * 28 * 28, num_classes)

    def forward(self, x1, x2):
        features1 = self.cnn_layers1(x1)
        features2 = self.cnn_layers2(x2)
        fused_features = self.attention_fusion(features1, features2)
        output = self.classifier(fused_features)
        return output