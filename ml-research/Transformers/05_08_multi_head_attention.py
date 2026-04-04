import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product attention.
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, V)
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    Q_reshaped = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_reshaped = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_reshaped = V_proj.reshape(batch_size, seq_len, num_heads, d_k)

    Q_transposed = Q_reshaped.transpose(0, 2, 1, 3)
    K_transposed = K_reshaped.transpose(0, 2, 1, 3)
    V_transposed = V_reshaped.transpose(0, 2, 1, 3)

    output = scaled_dot_product_attention(Q_transposed, K_transposed, V_transposed)
    output = output.transpose(0, 2, 1, 3)

    concat = output.reshape(batch_size, seq_len, d_model)
    result = concat @ W_o
    return result
