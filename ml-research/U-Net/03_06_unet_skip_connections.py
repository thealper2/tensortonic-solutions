import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features and concatenate with decoder features.
    """
    _, H_enc, W_enc, _ = encoder_features.shape
    _, H_dec, W_dec, _ = decoder_features.shape

    diff_H = H_enc - H_dec
    diff_W = W_enc - W_dec

    start_H = diff_H // 2
    start_W = diff_W // 2

    encoder_cropped = encoder_features[
        :,
        start_H:start_H + H_dec,
        start_W:start_W + W_dec,
        :
    ]

    output = np.concatenate([encoder_cropped, decoder_features], axis=-1)
    return output
