import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torch_geometric.io import read_off
import trimesh
import pyrender
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"

# Read the metadata file and create a global dictionary of classes
metadata_file = '../metadata_modelnet40.csv'
metadata = pd.read_csv(metadata_file)
class_names = metadata['class'].unique()
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

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

# class AttentionFusionLayer(nn.Module):
#     def __init__(self, input_dim):
#         super(AttentionFusionLayer, self).__init__()
#         self.fc1 = nn.Linear(input_dim * 2, input_dim)
#         self.fc2 = nn.Linear(input_dim, 2)  # Two weights, one for each modality

#     def forward(self, x1, x2):
#         # Combine the two modalities
#         combined = torch.cat((x1, x2), dim=1)  # Shape: (B, 128, 14, 14)
        
#         # Apply attention mechanism
#         attention_scores = self.fc2(F.relu(self.fc1(combined.mean(dim=[2, 3]))))  # Reduce spatial dimensions
#         attention_weights = F.softmax(attention_scores, dim=1)  # Shape: (B, 2)

#         # Apply attention weights to both modalities
#         fused_features = (
#             attention_weights[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x1
#             + attention_weights[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x2
#         )  # Shape: (B, 64, 14, 14)

#         # Flatten the spatial dimensions
#         fused_features = fused_features.view(fused_features.size(0), -1)  # Shape: (B, 64 * 14 * 14)
#         return fused_features

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
    def __init__(self, input_dims, output_dims, kernel_sizes, num_classes):  # Default values for input_size and num_classes
        super(CombinedModel, self).__init__()
        self.cnn_layers1 = CNNAutoencoder(input_dims, output_dims, kernel_sizes)
        # self.cnn_layers2 = CNNAutoencoder(input_dims, output_dims, kernel_sizes)
        # self.attention_fusion = AttentionFusionLayer(input_dim=output_dims[-1])
        self.classifier = Classifier(14 * 14 * 64, num_classes)

    def forward(self, x1, x2):
        features1 = self.cnn_layers1.encoder(x1)
        features1 = features1.view(features1.size(0), -1)
        # features2 = self.cnn_layers2.encoder(x2)
        # fused_features = self.attention_fusion(features1, features2)
        output = self.classifier(features1)
        return output

class CustomDataset(Dataset):
    def __init__(self, data_dir, tensor_dir, transform=None, image_size=(224, 224)):
        self.data_dir = data_dir
        self.tensor_dir = tensor_dir
        self.transform = transform
        self.image_size = image_size
        self.data = []
        for file_name in os.listdir(data_dir):
            self.data.append((os.path.join(data_dir, file_name), file_name.split("_")[0]))
        if len(self.data) == 0:
            raise ValueError(f"No .off files found in directory: {data_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, class_name = self.data[idx]
        tensor_file = os.path.join(self.tensor_dir, f"{os.path.basename(file_path)}.pt")
        if os.path.exists(tensor_file):
            modality1, modality2 = torch.load(tensor_file)
        else:
            print(f"Rendering and saving file: {file_path}")  # Debug print
            data = read_off(file_path)
            modality1, modality2 = self.generate_views(data)
            torch.save((modality1, modality2), tensor_file)
        label = class_to_idx[class_name]
        return modality1, modality2, label

    def generate_views(self, data):
        # Load the 3D shape using trimesh
        mesh = trimesh.Trimesh(vertices=data.pos.numpy(), faces=data.face.numpy().T)

        # Generate two different views
        view1 = self.render_view(mesh, angle=0)
        view2 = self.render_view(mesh, angle=90)
        return view1, view2

    def render_view(self, mesh, angle):
        # Create a scene and add the mesh
        scene = pyrender.Scene()
        mesh_node = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh_node)

        # Set up the camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = trimesh.transformations.rotation_matrix(
            np.radians(angle), [0, 1, 0]
        )[:3, :3]
        scene.add(camera, pose=camera_pose)

        # Render the scene
        renderer = pyrender.OffscreenRenderer(*self.image_size)
        color, _ = renderer.render(scene)
        renderer.delete()

        # Ensure the numpy array has positive strides
        color = np.ascontiguousarray(color)

        # Convert the rendered image to a tensor
        view = torch.tensor(color, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return view

def load_data(partition_id: int, num_partitions: int):
    """Load partition data from the specified data directory."""
    data_dir = f'../data/client{partition_id+1}'
    tensor_dir = f'../data/client{partition_id+1}/tensors'
    os.makedirs(tensor_dir, exist_ok=True)
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = CustomDataset(train_dir, tensor_dir, transform=pytorch_transforms)
    test_dataset = CustomDataset(test_dir, tensor_dir, transform=pytorch_transforms)

    # Reduce batch size to lower memory consumption
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)

    return trainloader, testloader

def train_feature_extractor(net, trainloader, epochs, device, modality_idx):
    """Train the feature extractor on the training set using reconstruction loss."""
    print(f"Training the feature extractor for modality {modality_idx}")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    count = 0
    for _ in range(epochs):
        for batch in trainloader:
            count += 1
            modality = batch[modality_idx].to(device)
            optimizer.zero_grad()
            features = net.encoder(modality)
            reconstruction = net.decoder(features)
            loss = criterion(reconstruction, modality)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {_} Training loss: {running_loss / count}")
    avg_trainloss = running_loss / count
    print(f"Average training loss: {avg_trainloss}")
    return avg_trainloss

def train_classifier(net, trainloader, epochs, device):
    """Train the classifier on the combined features from both modalities."""
    print("Training the classifier")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    net.train()
    running_loss = 0.0
    count = 0
    for _ in range(epochs):
        for batch in trainloader:
            count += 1
            modality1, modality2, labels = batch
            modality1, modality2, labels = modality1.to(device), modality2.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(modality1, modality2)
            # print("output: ", output.shape)
            # print("labels: ", labels.shape)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {_} Training loss: {running_loss / count}")

    avg_trainloss = running_loss / count
    return avg_trainloss

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available

    # Train feature extractors
    train_feature_extractor(net.cnn_layers1, trainloader, epochs, device, modality_idx=0)
    # train_feature_extractor(net.cnn_layers1, trainloader, epochs, device, modality_idx=1)

    # Freeze the feature extractors
    for param in net.cnn_layers1.parameters():
        param.requires_grad = False
    # for param in net.cnn_layers2.parameters():
    #     param.requires_grad = False

    # Train classifier and attention layer
    avg_trainloss = train_classifier(net, trainloader, 2 * epochs, device)
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set."""
    print("Testing the model")
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    count = 0
    with torch.no_grad():
        for batch in testloader:
            count += len(batch)
            modality1, modality2, labels = batch
            modality1, modality2, labels = modality1.to(device), modality2.to(device), labels.to(device)
            output = net(modality1, modality2)
            loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    return loss, accuracy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
trainloader, testloader = load_data(partition_id=0, num_partitions=1)

# Initialize model
input_dims = [3, 8, 16, 32]  # Example input dimensions
output_dims = [8, 16, 32, 64]  # Example output dimensions
kernel_sizes = [3, 3, 3, 3]  # Example kernel sizes
num_classes = 40  # Number of classes in ModelNet40
model = CombinedModel(input_dims=input_dims, output_dims=output_dims, kernel_sizes=kernel_sizes, num_classes=num_classes)

# Train model
epochs = 10
avg_trainloss = train(model, trainloader, epochs, device)
print(f"Average training loss: {avg_trainloss}")

# Test model
test_loss, test_accuracy = test(model, testloader, device)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
