import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class CNNAutoencoder(nn.Module):
    """A CNN autoencoder.

    Stacks n layers of (Conv2d, MaxPool2d, BatchNorm2d) for the encoder and
    (ConvTranspose2d, BatchNorm2d) for the decoder, where n is determined
    by the length of the input args.

    Args:
        input_dims (List[int]): List of input dimensions.
        output_dims (List[int]): List of output dimensions. Should match
            input_dims offset by one.
        kernel_sizes (List[int]): Kernel sizes for convolutions. Should match
            the sizes of cnn_input_dims and cnn_output_dims.

    Inputs:
        x (Tensor): Tensor containing a batch of images.
    """

    def __init__(
        self, input_dims: List[int], output_dims: List[int], kernel_sizes: List[int]
    ):
        super().__init__()
        assert len(input_dims) == len(output_dims) and len(output_dims) == len(
            kernel_sizes
        ), "input_dims, output_dims, and kernel_sizes should all have the same length"
        assert (
            input_dims[1:] == output_dims[:-1]
        ), "output_dims should match input_dims offset by one"

        # Encoder
        encoder_layers: List[nn.Module] = []
        for in_channels, out_channels, kernel_size in zip(
            input_dims,
            output_dims,
            kernel_sizes,
        ):
            padding_size = kernel_size // 2

            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding_size
            )

            max_pool2d = nn.MaxPool2d(2, stride=2)
            batch_norm_2d = nn.BatchNorm2d(out_channels)

            encoder_layers.append(
                nn.Sequential(conv, nn.LeakyReLU(), max_pool2d, batch_norm_2d)
            )

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        for in_channels, out_channels, kernel_size in zip(
            reversed(output_dims),
            reversed(input_dims),
            reversed(kernel_sizes),
        ):
            padding_size = kernel_size // 2

            deconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=2, padding=padding_size, output_padding=1
            )

            batch_norm_2d = nn.BatchNorm2d(out_channels)

            decoder_layers.append(
                nn.Sequential(deconv, nn.LeakyReLU(), batch_norm_2d)
            )

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AttentionFusionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusionLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, input_dim)
        self.fc2 = nn.Linear(input_dim, 2)  # Two weights, one for each modality

    def forward(self, x1, x2):
        # Combine the two modalities
        combined = torch.cat((x1, x2), dim=1)  # Shape: (B, 128, 14, 14)
        
        # Apply attention mechanism
        attention_scores = self.fc2(F.relu(self.fc1(combined.mean(dim=[2, 3]))))  # Reduce spatial dimensions
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: (B, 2)

        # Apply attention weights to both modalities
        fused_features = (
            attention_weights[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x1
            + attention_weights[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x2
        )  # Shape: (B, 64, 14, 14)

        # Flatten the spatial dimensions
        fused_features = fused_features.view(fused_features.size(0), -1)  # Shape: (B, 64 * 14 * 14)
        return fused_features

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        print("Classifier:")
        self.fc1 = nn.Linear(input_dim, 256)
        # print("fc1: ", input_dim, 64)
        self.fc2 = nn.Linear(256, num_classes)
        # print("fc2: ", 64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CombinedModel(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_sizes, num_classes):  # Default values for input_size and num_classes
        super(CombinedModel, self).__init__()
        self.cnn_layers1 = CNNAutoencoder(input_dims, output_dims, kernel_sizes)
        self.cnn_layers2 = CNNAutoencoder(input_dims, output_dims, kernel_sizes)
        self.attention_fusion = AttentionFusionLayer(input_dim=output_dims[-1])
        self.classifier = Classifier(14 * 14 * 64, num_classes)

    def forward(self, x1, x2):
        features1 = self.cnn_layers1.encoder(x1)
        features2 = self.cnn_layers2.encoder(x2)
        fused_features = self.attention_fusion(features1, features2)
        output = self.classifier(fused_features)
        return output

def main():
    import os
    input_dims = [3, 8, 16, 32]  # Example input dimensions
    output_dims = [8, 16, 32, 64]  # Example output dimensions
    kernel_sizes = [3, 3, 3, 3]  # Example kernel sizes
    num_classes = 40  # Number of classes in ModelNet40
    model = CombinedModel(input_dims=input_dims, output_dims=output_dims, kernel_sizes=kernel_sizes, num_classes=num_classes)
    tensor_file = '../data/client1/tensors/table_0330.pt'
    if os.path.exists(tensor_file):
        modality1, modality2 = torch.load(tensor_file)
    print("modality1: ", modality1.shape)
    print("modality2: ", modality2.shape)
    print("Inside main")
