import torch as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_accuracy(data_loader, device, model):
    """
    Calculate the accuracy of the model on the test dataset.

    Parameters:
    - data_loader: DataLoader for the test dataset
    - device: Device to perform computations on (CPU or GPU)
    - model: The model to evaluate

    Returns:
    - test_accuracy: Accuracy of the model on the test dataset
    - y_pred: List of predicted class labels
    - y_pred_2: List of raw model outputs
    """
    y_pred = []  # To store predicted class labels
    y_pred_2 = []  # To store raw outputs from the model
    test_accuracy = 0  # Initialize accuracy

    with nn.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in data_loader:
            inputs = inputs.to(device)  # Move inputs to the appropriate device
            labels = labels.to(device)  # Move labels to the appropriate device

            outputs = model(inputs)  # Forward pass through the model
            y_pred_2.extend(outputs)  # Store raw outputs
            _, predicted_labels = nn.max(outputs, dim=1)  # Get predicted class labels
            y_pred.extend(predicted_labels.tolist())  # Convert tensor to list and store

            # Count correctly predicted samples
            test_accuracy += (predicted_labels == labels).sum().item()  

    # Calculate test accuracy as a percentage
    test_accuracy = 100.0 * test_accuracy / len(data_loader.dataset)
    return test_accuracy, y_pred, y_pred_2

def train_model(model, lossFunction, optimizer, scheduler, train_loader, test_loader, num_epochs, validation=True, regularize=True):
    """
    Train the model for a specified number of epochs and evaluate its performance.

    Parameters:
    - model: The model to train
    - lossFunction: Loss function to optimize
    - optimizer: Optimizer for training
    - scheduler: Learning rate scheduler
    - train_loader: DataLoader for the training dataset
    - test_loader: DataLoader for the test/validation dataset
    - num_epochs: Number of epochs for training
    - validation: Boolean indicating whether to validate after each epoch
    - regularize: Boolean indicating whether to use learning rate scheduling

    Returns:
    - train_loss_history: History of training losses
    - train_acc_history: History of training accuracies
    - val_loss_history: History of validation losses
    - val_acc_history: History of validation accuracies
    """
    train_loss_history = []  # Store training loss history
    train_acc_history = []  # Store training accuracy history
    val_loss_history = []  # Store validation loss history
    val_acc_history = []  # Store validation accuracy history

    for epoch in range(num_epochs):
        train_loss = 0.0  # Initialize training loss
        train_correct = 0  # Initialize number of correct predictions
        train_total = 0  # Initialize total number of samples

        # Set the model to training mode
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # Move inputs to the appropriate device
            labels = labels.to(device)  # Move labels to the appropriate device
            
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(inputs)  # Forward pass through the model
            loss = lossFunction(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            
            # Update weights
            if regularize:
                scheduler.step()  # Step the scheduler if using regularization
            else:
                optimizer.step()  # Update model parameters

            train_loss += loss.item() * inputs.size(0)  # Accumulate training loss
            _, predicted = outputs.max(1)  # Get predicted class labels
            train_total += labels.size(0)  # Update total samples
            train_correct += predicted.eq(labels).sum().item()  # Count correct predictions

        # Calculate average training loss and accuracy
        average_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / train_total
        train_loss_history.append(average_train_loss)  # Store training loss
        train_acc_history.append(train_accuracy)  # Store training accuracy

        log = f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"

        if validation:
            # Evaluate on the validation set
            model.eval()  # Set the model to evaluation mode

            val_loss = 0.0  # Initialize validation loss
            val_correct = 0  # Initialize number of correct predictions
            val_total = 0  # Initialize total number of samples

            with nn.no_grad():  # Disable gradient calculation for evaluation
                for inputs, labels in test_loader:  # Iterate through validation dataset
                    inputs = inputs.to(device)  # Move inputs to the appropriate device
                    labels = labels.to(device)  # Move labels to the appropriate device

                    outputs = model(inputs)  # Forward pass through the model
                    loss = lossFunction(outputs, labels)  # Calculate loss

                    val_loss += loss.item() * inputs.size(0)  # Accumulate validation loss
                    _, predicted = outputs.max(1)  # Get predicted class labels
                    val_total += labels.size(0)  # Update total samples
                    val_correct += predicted.eq(labels).sum().item()  # Count correct predictions

            # Calculate average validation loss and accuracy
            average_val_loss = val_loss / len(test_loader.dataset)
            val_accuracy = 100.0 * val_correct / val_total
            val_loss_history.append(average_val_loss)  # Store validation loss
            val_acc_history.append(val_accuracy)  # Store validation accuracy

            # Log validation results
            log = str(log) + str(f", Val Loss: {average_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}")

        # Print training loss and accuracy for the current epoch
        print(log)

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history
