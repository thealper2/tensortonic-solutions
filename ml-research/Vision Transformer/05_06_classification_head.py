import numpy as np

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer normalization along the last axis."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    B, N, D = encoder_output.shape
    cls_token = encoder_output[:, 0]

    if W_head is None:
        W_head = np.random.randn(D, num_classes) * 0.02

    h_cls_hat = layer_norm(cls_token)
    logits = h_cls_hat @ W_head
    return logits