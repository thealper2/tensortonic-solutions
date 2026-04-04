import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
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

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)  # Fixed: use variance instead of mean
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    output = gamma * x_normalized + beta
    return output
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
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

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    batch_size, seq_len, d_model = x.shape
    z1 = np.dot(x, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU activation
    output = np.dot(a1, W2) + b2
    return output

# Alternative implementation with explicit shapes and debugging
def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                            W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                            b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                            gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Save input for residual connections
    residual = x
    
    # Layer normalization before attention
    x_norm = layer_norm(x, gamma1, beta1)
    
    # Multi-head attention (self-attention)
    attn_output = multi_head_attention(x_norm, x_norm, x_norm, 
                                       W_q, W_k, W_v, W_o, num_heads)
    
    # First residual connection
    x = residual + attn_output
    
    # Save for second residual
    residual = x
    
    # Layer normalization before FFN
    x_norm = layer_norm(x, gamma2, beta2)
    
    # Feed-forward network
    ff_output = feed_forward(x_norm, W1, b1, W2, b2)
    
    # Second residual connection
    output = residual + ff_output
    
    return output
