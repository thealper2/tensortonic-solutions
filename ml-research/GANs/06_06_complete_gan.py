import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.noise_dim = len(G_W)
        
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        logits = np.tanh(np.dot(z, self.G_W))
        return logits
    
    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        logits = np.dot(x, self.D_W)
        sigmoid = 1 / (1 + np.exp(-logits))
        return sigmoid
    
    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        clip_probs = lambda x: np.clip(x, 1e-8, 1-1e-8)
        
        fake_data = self.generate(z)
        
        r_logits = np.dot(real_data, self.D_W)
        f_logits = np.dot(fake_data, self.D_W)
        
        r_probs = sigmoid(r_logits)
        f_probs = sigmoid(f_logits)
        
        d_loss = -np.mean(np.log(clip_probs(r_probs)) + np.log(1 - clip_probs(f_probs)))
        g_loss = -np.mean(np.log(clip_probs(f_probs)))
        return { "d_loss": d_loss, "g_loss": g_loss }
