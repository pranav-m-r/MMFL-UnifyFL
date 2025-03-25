import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class CNNAutoencoder(nn.Module):
    """A CNN autoencoder with adjusted pooling to preserve spatial dimensions."""

    def __init__(self, input_dims: List[int], output_dims: List[int], kernel_sizes: List[int]):
        super().__init__()
        assert len(input_dims) == len(output_dims) and len(output_dims) == len(kernel_sizes), \
            "input_dims, output_dims, and kernel_sizes should all have the same length"
        assert input_dims[1:] == output_dims[:-1], "output_dims should match input_dims offset by one"

        # Encoder
        encoder_layers: List[nn.Module] = []
        for in_channels, out_channels, kernel_size in zip(input_dims, output_dims, kernel_sizes):
            padding_size = kernel_size // 2

            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding_size)
            # Use pooling only for the first two layers to preserve spatial dimensions
            max_pool2d = nn.MaxPool2d(2, stride=2) if in_channels < output_dims[-3] else nn.Identity()
            batch_norm_2d = nn.BatchNorm2d(out_channels)

            encoder_layers.append(nn.Sequential(conv, nn.LeakyReLU(), max_pool2d, batch_norm_2d))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        for in_channels, out_channels, kernel_size in zip(reversed(output_dims), reversed(input_dims), reversed(kernel_sizes)):
            padding_size = kernel_size // 2

            deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding_size, output_padding=1)
            batch_norm_2d = nn.BatchNorm2d(out_channels)

            decoder_layers.append(nn.Sequential(deconv, nn.LeakyReLU(), batch_norm_2d))

        self.decoder = nn.Sequential(*decoder_layers)

        # Final layer to ensure output matches input size
        self.final_layer = nn.Conv2d(input_dims[0], input_dims[0], kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        reconstructed = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        return reconstructed

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, outputs):
        # outputs: [batch_size, seq_len, hidden_dim]
        scores = self.attn(outputs).squeeze(-1)  # [batch_size, seq_len]
        attn_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len]
        context = torch.sum(attn_weights.unsqueeze(-1) * outputs, dim=1)
        return context

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class CombinedModel(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_sizes, num_classes):
        super(CombinedModel, self).__init__()
        self.cnn_layers1 = CNNAutoencoder(input_dims, output_dims, kernel_sizes)
        self.cnn_layers2 = CNNAutoencoder(input_dims, output_dims, kernel_sizes)
        self.attention = AttentionLayer(8)
        # Concatenation doubles the feature size, so adjust the input dimension for the classifier
        self.classifier = Classifier(2 * 64 * 8 * 8, num_classes)  # Include 7 * 7 for spatial dimensions

    def forward(self, x1, x2):
        features1 = self.cnn_layers1.encoder(x1)
        features2 = self.cnn_layers2.encoder(x2)
        # print(f"Features1 shape: {features1.shape}, Features2 shape: {features2.shape}")

        context1 = self.attention(features1)
        context2 = self.attention(features2)

        concatenated_features = torch.cat((features1, features2), dim=1)
        # print(f"Concatenated features shape: {concatenated_features.shape}")

        flattened_features = concatenated_features.view(concatenated_features.size(0), -1)
        # print(f"Flattened features shape: {flattened_features.shape}")

        output = self.classifier(flattened_features)
        # print(f"Output shape: {output.shape}")
        return output

def train_feature_extractor(net, trainloader, epochs, device, modality_idx):
    """Train the feature extractor on the training set using reconstruction loss."""
    print(f"Training the feature extractor for modality {modality_idx}")
    net.to(device)  # Move model to GPU if available
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    count = 0
    for _ in range(epochs):
        for images, labels in trainloader:
            count += 1
            modality = images.to(device)
            optimizer.zero_grad()
            reconstruction = net(modality)
            # print(f"Input size: {modality.size()}, Reconstructed size: {reconstruction.size()}")
            loss = criterion(reconstruction, modality)  # Ensure sizes match
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {_} Training loss: {running_loss / count}")
    avg_trainloss = running_loss / count
    print(f"Average training loss: {avg_trainloss}")
    return avg_trainloss

def train_classifier(net, trainloader, testloader, epochs, device):
    """Train the classifier on the combined features from both modalities."""
    print("Training the classifier")
    net.to(device)  # Move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            modality1, modality2 = images.to(device), images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = net(modality1, modality2)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate on validation set
        val_loss, val_accuracy = test(net, testloader, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader)}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy * 100}%")
    return running_loss / len(trainloader)

def train(net, trainloader, testloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available

    # Train feature extractors
    train_feature_extractor(net.cnn_layers1, trainloader, epochs, device, modality_idx=0)
    train_feature_extractor(net.cnn_layers2, trainloader, epochs, device, modality_idx=1)

    # Freeze the feature extractors
    for param in net.cnn_layers1.parameters():
        param.requires_grad = False
    for param in net.cnn_layers2.parameters():
        param.requires_grad = False

    # Train classifier and attention layer
    avg_trainloss = train_classifier(net, trainloader, testloader, 2 * epochs, device)
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set."""
    # print("Testing the model")
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            modality1, modality2 = images.to(device), images.to(device)
            labels = labels.to(device)
            output = net(modality1, modality2)
            loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    return loss, accuracy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to match MNIST dimensions
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32)

# Initialize model
input_dims = [3, 8, 16, 32]  # MNIST has 1 channel (grayscale)
output_dims = [8, 16, 32, 64]
kernel_sizes = [3, 3, 3, 3]
num_classes = 10  # MNIST has 10 classes (digits 0-9)
model = CombinedModel(input_dims=input_dims, output_dims=output_dims, kernel_sizes=kernel_sizes, num_classes=num_classes).to(device)

# Train model
epochs = 25
avg_trainloss = train(model, trainloader, testloader, epochs, device)
print(f"Average training loss: {avg_trainloss}")

# Test model
print("Testing the model")
test_loss, test_accuracy = test(model, testloader, device)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy * 100}%")