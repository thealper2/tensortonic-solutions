import numpy as np

def vgg_classifier(features: np.ndarray, num_classes: int = 1000) -> np.ndarray:
    """
    Implement VGG's fully connected classifier.
    """
    B = features.shape[0]
    
    # Flatten
    x = features.reshape(B, -1)  # (B, 25088)
    
    # FC1 -> (B, 4096)
    x = x[:, :4096]
    x = np.maximum(0, x)  # ReLU
    
    # FC2 -> (B, 4096)
    x = x[:, :4096]
    x = np.maximum(0, x)  # ReLU
    
    # FC3 -> (B, num_classes)
    logits = x[:, :num_classes]
    
    return logits
