import numpy as np
from scipy.interpolate import griddata

import constants


def interpolate(image):
    """ Interpolate the transmission image.
        Currently implemented as simple linear interpolation.
    """
    x, y = np.meshgrid(np.arange(0, image.shape[1], 1), np.arange(0, image.shape[0], 1))
    mesh_grid = (y, x)
    nonzero_loc = np.nonzero(image)
    nonzero_val = image[nonzero_loc]
    interpolated = griddata(nonzero_loc, nonzero_val, mesh_grid, method='linear', fill_value=1)

    return interpolated


def clip(image):
    """ Places a threshold on image values. """

    image[image > constants.max_transmission] = constants.max_transmission
    image[image < constants.min_transmission] = constants.min_transmission

    return image
