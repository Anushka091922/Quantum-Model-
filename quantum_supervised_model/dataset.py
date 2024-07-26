import os  # Import the os module for operating system functionalities
import urllib.request  # Import urllib.request for downloading files
import zipfile  # Import zipfile for extracting zip files
import torch  # Import PyTorch for deep learning functionalities
import medmnist  # Import medmnist for medical image datasets
import numpy as np  # Import NumPy for numerical operations
import torch as nn  # Import PyTorch's neural network module
import pandas as pd  # Import Pandas for data manipulation and analysis
import torchvision.transforms as transforms  # Import torchvision.transforms for image transformations
import torchvision.datasets as datasets  # Import torchvision.datasets for standard datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset  # Import PyTorch data utilities
from medmnist import INFO  # Import metadata info for medmnist datasets
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling

def load_dataset(input_shape, batch_size, data_dir, data_transforms):
    """
    Load the training and validation datasets using ImageFolder.
    
    Args:
        input_shape (tuple): The shape of the input images (channels, height, width).
        batch_size (int): Number of samples per batch.
        data_dir (str): Directory containing the training and validation data.
        data_transforms (dict): Dictionary of transformations for training and validation datasets.

    Returns:
        Tuple: train_dataset, test_dataset, train_loader, test_loader, dataset_sizes, class_names, y_train, y_test
    """
    
    # Create datasets for training and validation using ImageFolder
    image_datasets = {
        x if x == "train" else "val": datasets.ImageFolder(
            os.path.join(data_dir, x), data_transforms[x]  # Load images and apply transformations
        )
        for x in ["train", "val"]  # Iterate over training and validation folders
    }

    # Get dataset sizes for training and validation datasets
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes  # Get class names from the training dataset
    y_train = image_datasets["train"].targets  # Get targets (labels) for training data
    y_test = image_datasets["val"].targets  # Get targets (labels) for validation data

    # Initialize dataloaders for the datasets with specified batch size
    dataloaders = {
        x: nn.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
        for x in ["train", "val"]  # Create DataLoader for both training and validation datasets
    }

    train_dataset = image_datasets["train"]  # Reference to the training dataset
    test_dataset = image_datasets["val"]  # Reference to the validation dataset

    train_loader = dataloaders['train']  # DataLoader for training data
    test_loader = dataloaders['val']  # DataLoader for validation data

    # Print the size of the training and validation datasets
    print("Training dataset size: " + str(len(train_dataset)))
    print("Testing dataset size: " + str(len(test_dataset)))

    return train_dataset, test_dataset, train_loader, test_loader, dataset_sizes, class_names, y_train, y_test


def get_stats(train_loader, input_shape):
    """
    Calculate the mean and standard deviation of the training dataset.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        input_shape (tuple): The shape of the input images (channels, height, width).

    Returns:
        Tuple: total_mean, total_std
    """
    
    # Initialize tensors for cumulative sum and sum of squares of pixel values
    psum = nn.tensor([0.0, 0.0, 0.0])  # Cumulative sum of pixel values for each channel
    psum_sq = nn.tensor([0.0, 0.0, 0.0])  # Cumulative sum of squared pixel values for each channel

    # Loop through the images in the training set to calculate cumulative sums
    for inputs, labels in train_loader:
        psum += inputs.sum(axis=[0, 2, 3])  # Sum over channels, height, and width for pixel values
        psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])  # Sum of squares of pixel values

    ####### FINAL CALCULATIONS #######

    # Calculate the total pixel count for averaging
    count = len(train_loader.dataset) * input_shape[0] * input_shape[0]

    # Calculate mean pixel values for each channel
    total_mean = psum / count  # Mean = total sum / number of pixels
    total_var = (psum_sq / count) - (total_mean ** 2)  # Variance = mean of squares - square of mean
    total_std = nn.sqrt(total_var)  # Standard deviation = square root of variance

    # Output the calculated mean and standard deviation
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))

    return total_mean, total_std


def download_and_extract_data(url, destination_folder):
    """
    Download a zip file from a URL and extract its contents to a specified folder.

    Args:
        url (str): URL of the zip file to download.
        destination_folder (str): Folder where the contents will be extracted.
    """
    
    # Ensure the destination folder exists; create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)  # Create the directory if it doesn't exist

    # File path to save the downloaded zip file
    zip_file_path = os.path.join(destination_folder, os.path.basename(url))

    try:
        # Download the zip file from the specified URL
        urllib.request.urlretrieve(url, zip_file_path)

        # Extract the contents of the downloaded zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)  # Extract to the specified folder

        # Remove the downloaded zip file after extraction to save space
        os.remove(zip_file_path)

        print(f"Data downloaded and extracted to '{destination_folder}' folder.")
    except Exception as e:
        print(f"Error: {e}")  # Print any errors that occur during download/extraction


class SmallDataset(Dataset):
    """
    A custom Dataset class that creates a smaller dataset with a specified number of samples per class.
    """
    
    def __init__(self, samples_per_class, dataset):
        """
        Initialize the SmallDataset instance.

        Args:
            samples_per_class (int): Number of samples to take from each class.
            dataset (Dataset): The original dataset.
        """
        
        self.samples_per_class = samples_per_class  # Store the number of samples per class
        self.dataset = dataset  # Store the reference to the original dataset
        self.class_indices = self._get_class_indices()  # Get indices of samples for each class

    def _get_class_indices(self):
        """
        Create a dictionary mapping class labels to their indices in the dataset.

        Returns:
            dict: A dictionary with class labels as keys and lists of indices as values.
        """
        
        class_indices = {}  # Initialize an empty dictionary to hold class indices
        for idx, (_, label_arr) in enumerate(self.dataset):
            label = label_arr[0]  # Get the class label for the current sample
            if label not in class_indices:
                class_indices[label] = []  # Create a new entry for the class if it doesn't exist
            class_indices[label].append(idx)  # Add the index of the sample to the class's list of indices
        return class_indices  # Return the dictionary of class indices

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples (samples_per_class * number of classes).
        """
        
        return len(self.class_indices) * self.samples_per_class  # Return the total count of samples

    def __getitem__(self, idx):
        """
        Get a sample and its corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple: A tuple containing the sample and its label.
        """
        
        class_idx = idx // self.samples_per_class  # Determine which class this sample belongs to
        label = list(self.class_indices.keys())[class_idx]  # Get the class label based on the index
        indices = self.class_indices[label]  # Get the list of indices for the selected class
        selected_idx = indices[idx % self.samples_per_class]  # Select an index for this sample
        return self.dataset[selected_idx]  # Return the sample from the original dataset


def create_small_dataset(samples_per_class, dataset, batch_size):
    """
    Create a small dataset with a specified number of samples per class.

    Args:
        samples_per_class (int): Number of samples to take from each class.
        dataset (Dataset): The original dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: A DataLoader for the small dataset.
    """
    
    # Get the indices of samples for each class from the original dataset
    class_indices = {}
    for idx, (_, label_arr) in enumerate(dataset):
        label = label_arr[0]  # Get the class label for the current sample
        if label not in class_indices:
            class_indices[label] = []  # Create a new entry for the class if it doesn't exist
        class_indices[label].append(idx)  # Add the index of the sample to the class's list of indices

    # Select samples from each class based on samples_per_class
    selected_indices = []
    for label, indices in class_indices.items():
        selected_indices.extend(indices[:samples_per_class])  # Select samples_per_class indices for each class

    # Create a SubsetRandomSampler from the selected indices to ensure random sampling
    sampler = SubsetRandomSampler(selected_indices)

    # Create a new DataLoader with the custom sampler and specified batch size
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def get_data_class(data_flag):
    """
    Get the data class corresponding to the specified flag.

    Args:
        data_flag (str): Identifier for the dataset (e.g., 'breastmnist').

    Returns:
        DataClass: The corresponding data class from the medmnist library.
    """
    
    download = True  # Flag to determine if data should be downloaded

    info = INFO[data_flag]  # Get metadata for the specified dataset using the data_flag
    task = info['task']  # Task type (e.g., classification)
    n_channels = info['n_channels']  # Number of input channels for the images
    n_classes = len(info['label'])  # Number of output classes based on the labels

    # Dynamically retrieve the data class from medmnist based on the metadata
    DataClass = getattr(medmnist, info['python_class'])  
    return DataClass  # Return the retrieved data class


def convert_2048_features(train_loader, device, encoder):
    """
    Convert images from the training loader to 2048-dimensional feature vectors using a pre-trained encoder.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        device (str): Device to perform calculations on ('cpu' or 'cuda').
        encoder (nn.Module): Pre-trained model to extract features.

    Returns:
        Tuple: x_2048_train, y_2048_train
    """
    
    X_2048_train = []  # List to hold extracted feature vectors for training data
    y_2048_train = []  # List to hold corresponding labels for the feature vectors
    with torch.no_grad():  # Disable gradient calculation to save memory and computation
        # Iterate through the batches of the training data
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # Move input images to the specified device (CPU or GPU)
            features = encoder(inputs)  # Extract features from the images using the encoder
            X_2048_train.append(features.cpu().numpy())  # Append the extracted features to the list (convert to NumPy)
            y_2048_train.append(labels.cpu().numpy())  # Append the corresponding labels to the list (convert to NumPy)

    # Concatenate all extracted features and labels into single arrays
    X_2048_train = np.concatenate(X_2048_train, axis=0)  # Combine all feature arrays into one
    y_2048_train = np.concatenate(y_2048_train, axis=0)  # Combine all label arrays into one

    return X_2048_train, y_2048_train  # Return the feature and label arrays


def scale_dataset(dataset):
    scaler = StandardScaler()
    scaled_dataset = scaler.fit_transform( dataset )
    return scaled_dataset

def create_new_dataframe_and_save_csv(X, y, num_cols, models_dir, file_name):
    merged_np_train_1 = np.concatenate((X, y), axis=1)

    merged_df_1 = pd.DataFrame(merged_np_train_1, columns =['f'+str(i) for i in range(num_cols)]+['y'])
    merged_df_1.to_csv(f'{models_dir}/{file_name}.csv')
    return merged_df_1
