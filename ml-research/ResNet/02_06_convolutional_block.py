import numpy as np

def relu(x):
    return np.maximum(0, x)

class ConvBlock:
    """
    Convolutional Block with projection shortcut.
    """
   
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
       
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01
       
        self.Ws = None
        if in_channels != out_channels or stride != 1:
            self.Ws = np.random.randn(in_channels, out_channels) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with projection shortcut when dimensions change.
        """
        h = relu(np.dot(x, self.W1))
        z = np.dot(h, self.W2)
        
        if self.Ws is not None:
            shortcut = np.dot(x, self.Ws)
        else:
            shortcut = x
        
        y = relu(z + shortcut)
        return y
