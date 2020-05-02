import autograd.numpy as np


def recover(image, transmission, airlight):
    """ Recovers pixels without their airlight component. """

    alpha = np.ones(transmission.shape) - transmission
    for c in range(3):
        ac = np.multiply(alpha, airlight[c])
        iac = image[:, :, c] - ac
        image[:, :, c] = np.divide(iac, transmission)

    return image
