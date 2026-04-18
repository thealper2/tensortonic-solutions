import numpy as np

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer normalization along the last axis."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention."""

    head_dim = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    B, N, D = x.shape
    head_dim = D // num_heads

    if Wq is None:
        Wq = np.random.randn(D, D) * 0.02
    if Wk is None:
        Wk = np.random.randn(D, D) * 0.02
    if Wv is None:
        Wv = np.random.randn(D, D) * 0.02
    if Wo is None:
        Wo = np.random.randn(D, D) * 0.02

    x_norm = layer_norm(x)

    Q = x_norm @ Wq
    K = x_norm @ Wk
    V = x_norm @ Wv

    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    attn_output = scaled_dot_product_attention(Q, K, V)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, N, D)
    attn_output = attn_output @ Wo

    x = x + attn_output
    x_norm = layer_norm(x)
    hidden_dim = int(embed_dim * mlp_ratio)

    if W1 is None:
        W1 = np.random.randn(D, D) * 0.02
    if W2 is None:
        W2 = np.random.randn(D, D) * 0.02

    mlp_output = x_norm @ W1
    mlp_output = gelu(mlp_output)
    mlp_output = mlp_output @ W2

    output = x + mlp_output
    return output