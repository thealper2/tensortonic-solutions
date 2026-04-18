import numpy as np

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer normalization along last axis."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax with numerical stability."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention."""
    head_dim = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, V)

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """
        Initialize Vision Transformer. If weight arrays are provided, use them;
        otherwise initialize randomly.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        
        # Patch embedding dimensions
        patch_dim = patch_size * patch_size * 3  # RGB images have 3 channels
        
        # Initialize weights in order:
        # 1. Patch projection
        if W_patch is None:
            self.W_patch = np.random.randn(patch_dim, embed_dim) * 0.02
        else:
            self.W_patch = W_patch
        
        # 2. CLS token (1, 1, embed_dim)
        if cls_token is None:
            self.cls_token = np.random.randn(1, 1, embed_dim) * 0.02
        else:
            self.cls_token = cls_token
        
        # 3. Position embeddings (1, num_patches + 1, embed_dim)
        if pos_embed is None:
            self.pos_embed = np.random.randn(1, self.num_patches + 1, embed_dim) * 0.02
        else:
            self.pos_embed = pos_embed
        
        # 4. Encoder block weights for each layer
        if encoder_weights is None:
            self.encoder_weights = []
            for _ in range(depth):
                layer_weights = {
                    'Wq': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'Wk': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'Wv': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'Wo': np.random.randn(embed_dim, embed_dim) * 0.02,
                }
                # MLP weights
                hidden_dim = int(embed_dim * mlp_ratio)
                layer_weights['W1'] = np.random.randn(embed_dim, hidden_dim) * 0.02
                layer_weights['W2'] = np.random.randn(hidden_dim, embed_dim) * 0.02
                self.encoder_weights.append(layer_weights)
        else:
            self.encoder_weights = encoder_weights
        
        # 5. Classification head
        if W_head is None:
            self.W_head = np.random.randn(embed_dim, num_classes) * 0.02
        else:
            self.W_head = W_head
    
    def patch_embed(self, x: np.ndarray) -> np.ndarray:
        """Convert image to patch embeddings."""
        B, H, W, C = x.shape
        P = self.patch_size
        
        # Extract patches: (B, H//P, P, W//P, P, C)
        patches = x.reshape(B, H//P, P, W//P, P, C)
        # Rearrange: (B, H//P, W//P, P, P, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        # Flatten patches: (B, N, patch_dim)
        patches = patches.reshape(B, self.num_patches, -1)
        # Project: (B, N, embed_dim)
        embeddings = patches @ self.W_patch
        return embeddings
    
    def encoder_block(self, x: np.ndarray, layer_weights: dict) -> np.ndarray:
        """Single encoder block with Pre-LayerNorm."""
        B, N, D = x.shape
        num_heads = self.num_heads
        head_dim = D // num_heads
        
        # Step 1: LayerNorm before attention
        x_norm = layer_norm(x)
        
        # Step 2: Multi-Head Self-Attention
        Q = x_norm @ layer_weights['Wq']
        K = x_norm @ layer_weights['Wk']
        V = x_norm @ layer_weights['Wv']
        
        # Reshape for multi-head
        Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Attention
        attn = scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads
        attn = attn.transpose(0, 2, 1, 3).reshape(B, N, D)
        
        # Output projection
        attn = attn @ layer_weights['Wo']
        
        # Step 3: First residual
        x = x + attn
        
        # Step 4: LayerNorm before MLP
        x_norm = layer_norm(x)
        
        # Step 5: MLP with GELU
        mlp = x_norm @ layer_weights['W1']
        mlp = gelu(mlp)
        mlp = mlp @ layer_weights['W2']
        
        # Step 6: Second residual
        output = x + mlp
        
        return output
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        B = x.shape[0]
        
        # Step 1: Patch Embedding
        z = self.patch_embed(x)  # (B, N, D)
        
        # Step 2: Prepend [CLS] token
        cls_tokens = np.tile(self.cls_token, (B, 1, 1))  # (B, 1, D)
        z = np.concatenate([cls_tokens, z], axis=1)  # (B, N+1, D)
        
        # Step 3: Add position embeddings
        z = z + self.pos_embed  # (B, N+1, D)
        
        # Step 4: Apply encoder blocks
        for i in range(self.depth):
            z = self.encoder_block(z, self.encoder_weights[i])
        
        # Step 5: Extract [CLS] token and apply LayerNorm
        cls_output = z[:, 0, :]  # (B, D)
        cls_output = layer_norm(cls_output)
        
        # Step 6: Classification head
        logits = cls_output @ self.W_head  # (B, num_classes)
        
        return logits