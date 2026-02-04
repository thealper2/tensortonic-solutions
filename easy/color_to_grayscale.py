def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    grayscale = []
    for row in image:
        gray_row = []
        for pixel in row:
            r, g, b = pixel
            y = 0.299 * r + 0.587 * g + 0.114 * b
            gray_row.append(y)

        grayscale.append(gray_row)

    return grayscale
