"""mmfl: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torch_geometric.io import read_off
import os


# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        for file_name in os.listdir(data_dir):
            self.data.append((os.path.join(data_dir, file_name), file_name.split("_")[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, class_name = self.data[idx]
        # data = read_off(file_path)
        data = read_off(file_path)
        modality1, modality2 = data.pos, data.face
        # if self.transform:
        #     modality1 = self.transform(modality1)
        #     modality2 = self.transform(modality2)
        return modality1, modality2, class_name
    

def load_data(partition_id: int, num_partitions: int):
    """Load partition data from the specified data directory."""
    data_dir = f'..\\data\\client{partition_id+1}'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = CustomDataset(train_dir, transform=pytorch_transforms)
    test_dataset = CustomDataset(test_dir, transform=pytorch_transforms)

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)

    return trainloader, testloader


def train_feature_extractor(net, trainloader, epochs, device, modality_idx):
    """Train the feature extractor on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            modality = batch[modality_idx]
            labels = batch[2]
            optimizer.zero_grad()
            loss = criterion(net(modality.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def train_classifier(net, trainloader, epochs, device):
    """Train the classifier on the combined features from both modalities."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            modality1, modality2, labels = batch
            optimizer.zero_grad()
            outputs = net(modality1.to(device), modality2.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    feature_extractor1, feature_extractor2, classifier = net.feature_extractor1, net.feature_extractor2, net.classifier

    # Train feature extractors
    train_feature_extractor(feature_extractor1, trainloader, epochs, device, modality_idx=0)
    train_feature_extractor(feature_extractor2, trainloader, epochs, device, modality_idx=1)

    # Train classifier
    avg_trainloss = train_classifier(classifier, trainloader, epochs, device)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            modality1, modality2, labels = batch
            outputs = net(modality1.to(device), modality2.to(device))
            loss += criterion(outputs, labels.to(device)).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)