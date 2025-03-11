import csv
import os
import shutil

NUM_CLIENTS = 5
DATA_DIR = '.\\data'

# Create client directories
for i in range(1, NUM_CLIENTS + 1):
    os.makedirs(os.path.join(DATA_DIR, f'client{i}\\train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, f'client{i}\\test'), exist_ok=True)

# Move files based on metadata
i = 1
with open('metadata_modelnet40.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        source_path = os.path.join('ModelNet40', row['object_path'].replace('/', '\\'))  # Ensure this matches the CSV column name
        filename = row['object_id']  # Ensure this matches the CSV column name
        client_dir = os.path.join(DATA_DIR, f'client{(i % NUM_CLIENTS) + 1}\\{row['split']}')
        destination_path = os.path.join(client_dir, filename)

        if os.path.exists(source_path):  # Check if the source file exists before moving
            shutil.move(source_path, destination_path)
        else:
            print(f"Warning: {source_path} not found.")

        i += 1
