import torch.nn as nn
from torchvision import models

def get_model(num_classes=3, model_name='resnet18', pretrained=True, dropout=0.5):
    """
    Get a pretrained model with custom classifier head.
    
    Args:
        num_classes: Number of output classes
        model_name: Model architecture ('resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'densenet121')
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability for the classifier head
    
    Returns:
        Model with custom classifier
    """
    if model_name.startswith('resnet'):
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown ResNet variant: {model_name}")
        
        num_ftrs = model.fc.in_features
        # Replace with dropout and new classifier
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )
    
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )
    
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )
    
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        num_ftrs = model.classifier[3].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    else:
        # Default to ResNet18
        print(f"Unknown model {model_name}, using ResNet18")
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )
    
    return model

