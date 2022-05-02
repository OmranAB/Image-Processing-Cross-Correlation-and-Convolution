import numpy as np


# image dimensions = n1 x n2
# filter dimensions = f1 x f2
# Cross-correlation operation implemented for three modes ('valid', 'same', 'full') affecting the size of the resulted image.
def cross_correlation(image, filter, mode='valid'):
    result_image = 0
    # Call numpy.shape to return a tuple of array's lengths according to the dimensions.
    # Assign image dimensions to n1 and n2.
    n1 = image.shape[0]
    n2 = image.shape[1]
    # Assign filter dimensions to f1 and f2.
    f1 = filter.shape[0]
    f2 = filter.shape[1]
    # valid mode dimensions of resulted image = (n1 - f1 + 1) x (n2 - f2 + 1)
    if(mode == 'valid'):
        dimension1 = n1 - f1 + 1
        dimension2 = n2 - f2 + 1
        result_image = np.empty((dimension1, dimension2))
        # Iterate over the list of the resulted image filling the cells with the resulted value of applying the filter.
        for i in range(dimension1):
            for j in range(dimension2):
                result_image[i][j] = np.sum(image[i:f1 + i, j:f2 + j]*filter)

    # same mode dimensions of resulted image = n1 x n2 (pad outer parts of the image with 0s to centralize the filter at the edges)
    if(mode == 'same'):
        result_image = np.empty((n1, n2))
        xAxis = f1 // 2
        yAxis = f2 // 2
        image_pad = np.pad(image, ((xAxis, xAxis), (yAxis, yAxis)),
                           'constant', constant_values=(0))
        for i in range(n1):
            for j in range(n2):
                result_image[i][j] = np.sum(
                    image_pad[i:f1 + i, j:f2 + j]*filter)
    # full mode dimensions of resulted image = (n1 + f2 - 1) x (n1 + f2 - 1)
    if(mode == 'full'):
        dimension1 = n1 + f1 - 1
        dimension2 = n2 + f2 - 1
        result_image = np.empty((dimension1, dimension2))
        image_pad = np.pad(image, ((f1 - 1, f1 - 1), (f2 - 1, f2 - 1)),
                           'constant', constant_values=(0))
        for i in range(dimension1):
            for j in range(dimension2):
                result_image[i][j] = np.sum(
                    image_pad[i:f1 + i, j:f2 + j]*filter)
    return np.array(result_image)

# Convolution operation using the implemented cross-correlation method and the numpy flip method to pass a fliped filter.


def convolution(image, filter, mode='valid'):
    return cross_correlation(image, np.flip(filter), mode)
