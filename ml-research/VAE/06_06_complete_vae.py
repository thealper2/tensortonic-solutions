import numpy as np

class VAE:
    def __init__(self, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray):
        """
        Initialize VAE with concrete weight matrices.
        """
        self.input_dim = W_mu.shape[0]
        self.latent_dim = W_mu.shape[1]
        
        self.W_mu = W_mu
        self.b_mu = b_mu
        self.W_logvar = W_logvar
        self.b_logvar = b_logvar
        self.W_dec = W_dec
        self.b_dec = b_dec

    def encode(self, x: np.ndarray) -> tuple:
        """
        Encode input to mean and log variance.
        """
        mu = x @ self.W_mu + self.b_mu
        log_var = x @ self.W_logvar + self.b_logvar
        return mu, log_var

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick.
        """
        sigma = np.exp(0.5 * log_var)
        z = mu + sigma * epsilon
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vector to reconstruction.
        """
        recon = z @ self.W_dec + self.b_dec
        return recon
    
    def forward(self, x: np.ndarray, epsilon: np.ndarray) -> dict:
        """
        Full forward pass: encode -> reparameterize -> decode.
        Returns dict with "recon", "mu", "log_var".
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, epsilon)
        recon = self.decode(z)
        return { "recon": recon, "mu": mu, "log_var": log_var }
    
    def generate(self, z: np.ndarray) -> np.ndarray:
        """
        Generate samples from given latent vectors.
        """
        samples = self.decode(z)
        return samples
