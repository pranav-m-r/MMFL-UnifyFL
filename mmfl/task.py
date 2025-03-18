from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torch_geometric.io import read_off
from typing import OrderedDict
import os
import pandas as pd
import torch
import trimesh
import pyrender
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"

# Read the metadata file and create a global dictionary of classes
metadata_file = '../metadata_modelnet40.csv'
metadata = pd.read_csv(metadata_file)
class_names = metadata['class'].unique()
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
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
    train_feature_extractor(net.cnn_layers2, trainloader, epochs, device, modality_idx=1)

    # Freeze the feature extractors
    for param in net.cnn_layers1.parameters():
        param.requires_grad = False
    for param in net.cnn_layers2.parameters():
        param.requires_grad = False

    # Train classifier and attention layer
    avg_trainloss = train_classifier(net, trainloader, epochs, device)
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

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)