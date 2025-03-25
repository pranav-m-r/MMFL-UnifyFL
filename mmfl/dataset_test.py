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

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

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
            modality1 = modality1.view(modality1.size(0), -1)
            output = net(modality1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {_} Training loss: {running_loss / count}")

    avg_trainloss = running_loss / count
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
            modality1 = modality1.view(modality1.size(0), -1)
            output = net(modality1)
            loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    return loss, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
trainloader, testloader = load_data(partition_id=0, num_partitions=1)

num_classes = 40  # Number of classes in ModelNet40
input_dims = 150528
model = Classifier(input_dim=input_dims, num_classes=num_classes)

# Train model
epochs = 50
avg_trainloss = train_classifier(model, trainloader, epochs, device)
print(f"Average training loss: {avg_trainloss}")

# Test model
test_loss, test_accuracy = test(model, testloader, device)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
