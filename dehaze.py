import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ransac import ransac
from window import SlidingWindow
import pickle
from scipy.interpolate import griddata


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


def recover(image, transmission, airlight):
    """ Recovers pixels without their airlight component. """

    alpha = np.ones(transmission.shape) - transmission
    for c in range(3):
        ac = np.multiply(alpha, airlight[c])
        iac = image[:, :, c] - ac
        image[:, :, c] = np.divide(iac, transmission)

    return image


def dehaze(image_path, airlight=np.array([0.9, 0.97, 0.988])):
    """ Dehazes an image using color-lines and a given airlight vector. """
    img = cv2.imread(image_path) / 255
    transmission_image = np.zeros(img.shape[:2])
    sliding_window = SlidingWindow(img)

    for window in sliding_window:
        patch = window.patch
        color_line = ransac(patch, iterations=3)

        if color_line.valid(airlight=airlight):
            transmission_image[window.y, window.x] = color_line.transmission

    transmission_image = interpolate(transmission_image)
    img = recover(img, transmission_image, airlight)
    return img


def main():
    dehazed = dehaze('bricks.png')
    plt.imshow(dehazed)
    plt.show()


if __name__ == '__main__':
    main()
