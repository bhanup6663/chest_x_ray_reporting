import os
from skimage import io
import torch
import torchxrayvision as xrv
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
import joblib


# Function to load the pre-trained DenseNet model
def load_densenet_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model

transform = transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])

# Function to preprocess and extract features from an image using DenseNet
def extract_features_from_image(image_path, model):
    # Load and preprocess the image
    img = io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)  # Normalize the image

    # If the image is already 2D (grayscale), skip channel conversion
    if len(img.shape) == 2:
        img = img[None, ...]  # Add a dummy channel to make it 3D (1, height, width)
    elif len(img.shape) == 3 and img.shape[2] == 3:  # If it's RGB
        img = img.mean(2)[None, ...]  # Convert to single color channel

    # Apply transformations (resize, crop, etc.)
    img = transform(img)

    # Convert to torch tensor and add batch dimension
    img_tensor = torch.from_numpy(img).unsqueeze(0)

    # Extract features using DenseNet model
    with torch.no_grad():
        features = model.features(img_tensor)  # Extract features
        features = features.cpu().detach().numpy().flatten()  # Convert to numpy and flatten the features
    
    return features

# Function to load the PCA model
def load_pca_model(pca_model_path):
    return joblib.load(pca_model_path)

# Function to load the KMeans model
def load_kmeans_model(kmeans_model_path):
    return joblib.load(kmeans_model_path)

# Function to process a list of images and predict clusters
def assign_clusters_to_images(image_path, densenet_model, pca_model, kmeans_model):
    
    # Extract visual features from the image using DenseNet
    visual_features = extract_features_from_image(image_path, densenet_model)
    
    # Perform PCA transformation
    reduced_features = pca_model.transform(visual_features.reshape(1, -1))
    
    # Predict the cluster for the given image
    cluster_prediction = kmeans_model.predict(reduced_features)[0]

    return cluster_prediction
        
        
# Main function to execute the inference process
def main(image_path, kmeans_model_path, pca_model_path):
    # Load models
    densenet_model = load_densenet_model()
    pca_model = load_pca_model(pca_model_path)
    kmeans_model = load_kmeans_model(kmeans_model_path)

    # Assign clusters to the new images
    cluster_prediction = assign_clusters_to_images(image_path, densenet_model, pca_model, kmeans_model)
    return cluster_prediction

if __name__ == "__main__":
    kmeans_model_path = "../../kmeans_models/kmeans_20_clusters.pkl"  
    pca_model_path = "../../kmeans_models/pca_model.pkl"  
    image_path = "../../test_resized/0af8628f5cbe0786db483a10934d1be5.png"
    print(main(image_path, kmeans_model_path, pca_model_path))
