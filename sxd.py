import os
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
from sklearn.cluster import KMeans
import shutil

# Check if CUDA is available and use it for computation if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ResNet50 model
model = models.resnet50(pretrained=True)
model.to(device)
model.eval()

def extract_features(image_paths):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_list = []
    for path in image_paths:
        img = Image.open(path)
        img_tensor = transform(img)
        img_tensor = torch.unsqueeze(img_tensor, 0).to(device)
        with torch.no_grad():
            features = model(img_tensor)
        features = features.squeeze().cpu().numpy()
        features_list.append(features)
    return np.array(features_list)

def kmeans_clustering(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(features)
    return predicted_labels

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            # print(f"Directory '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Error creating directory '{folder_path}': {e}")
    else:
        print(f"Directory '{folder_path}' already exists.")

def copy_file(source_file, destination_folder):
    # Check if the source file exists
    if not os.path.exists(source_file):
        print(f"Error: Source file '{source_file}' not found.")
        return

    # Check if the destination folder exists and create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get the filename from the source file
    filename = os.path.basename(source_file)

    # Create the destination path by joining the destination folder and filename
    destination_path = os.path.join(destination_folder, filename)

    try:
        # Copy the file to the destination path
        shutil.copy2(source_file, destination_path)
        print(f"File '{filename}' copied to '{destination_folder}' successfully.")
    except Exception as e:
        print(f"Error copying file: {e}")

def main():
    main_folder_path = "E:\\person_image\\ANYUJIN"  # Replace with the main folder path containing subfolders with images
    main_copy_path = "E:\\person_image_copy\\ANYUJIN"

    subfolders = os.listdir(main_folder_path)
    for subfolder in subfolders:
        folder_path = os.path.join(main_folder_path, subfolder)
        copy_path = os.path.join(main_copy_path, subfolder)

        image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

        # Extract features using ResNet50
        features = extract_features(image_paths)

        # Set the number of clusters for KMeans
        num_clusters = 5

        # Apply KMeans clustering on the features
        predicted_labels = kmeans_clustering(features, num_clusters)

        # Copy images to the corresponding cluster folders
        for label, path in zip(predicted_labels, image_paths):
            create_folder_if_not_exists(os.path.join(copy_path, str(label)))
            copy_file(path, os.path.join(copy_path, str(label)))

if __name__ == "__main__":
    main()
