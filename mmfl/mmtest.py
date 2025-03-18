from models import CombinedModel
from task import load_data, train, test
import torch

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
epochs = 20
avg_trainloss = train(model, trainloader, epochs, device)
print(f"Average training loss: {avg_trainloss}")

# Test model
test_loss, test_accuracy = test(model, testloader, device)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")