import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm.auto import tqdm

def evaluate_model(model, test_loader, device, class_names=None):
    """
    Evaluate model on test set and return predictions, true labels, and metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to run evaluation on
        class_names: List of class names for reporting
    
    Returns:
        Dictionary with predictions, true labels, accuracy, and classification report
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    
    # Classification report
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(all_labels)))]
    
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'predictions': np.array(all_preds),
        'true_labels': np.array(all_labels),
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }

def plot_confusion_matrix(cm, class_names, figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix from sklearn
        class_names: List of class names
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt.gcf()



