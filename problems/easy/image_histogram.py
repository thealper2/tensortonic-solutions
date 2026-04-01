def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    hist = [0] * 256
    for r in image:
        for pixel in r:
            hist[pixel] += 1

    return hist
