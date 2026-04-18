import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    B, H, W, C = image.shape

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    N = num_patches_h * num_patches_w

    patch_dim = patch_size * patch_size * C
    patches = image.reshape(B, num_patches_h, patch_size, num_patches_w, patch_size, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(B, N, patch_dim)

    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim) * 0.02

    embeddings = patches @ W_proj
    return embeddings
