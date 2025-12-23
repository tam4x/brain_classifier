import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

def get_train_transform(img_size=224, augmentation_type='standard'):
    """
    Get training transforms with different augmentation strategies.
    
    Args:
        img_size: Target image size
        augmentation_type: 'standard', 'aggressive', 'autoaugment', 'minimal'
    
    Returns:
        Composed transform
    """
    base_transforms = [
        transforms.Resize((img_size, img_size)),
    ]
    
    if augmentation_type == 'minimal':
        # Minimal augmentation - just basic transforms
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ]
    
    elif augmentation_type == 'standard':
        # Standard augmentation - balanced approach
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    elif augmentation_type == 'aggressive':
        # Aggressive augmentation - more transformations
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # Sometimes useful for medical images
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    elif augmentation_type == 'autoaugment':
        # AutoAugment - learned augmentation policy
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    else:
        # Default to standard if unknown type
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(base_transforms + augmentation_transforms)

def get_val_transform(img_size=224):
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        img_size: Target image size
    
    Returns:
        Composed transform
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

