import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224) -> np.ndarray:
    """Extract a random crop from the image."""
    height, width, channel = image.shape
    top = np.random.randint(0, height - crop_size + 1)
    left = np.random.randint(0, width - crop_size + 1)
    cropped = image[top:top+crop_size, left:left+crop_size, :]
    return cropped

def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Randomly flip image horizontally."""
    if np.random.rand() < p:
        return np.fliplr(image)

    return image
