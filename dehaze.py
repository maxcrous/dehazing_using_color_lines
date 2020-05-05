from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np

from linear_interpolation import clip
from linear_interpolation import interpolate
# from markov_random_field import mrf_interpolate
from ransac import ransac
from recover import recover
from window import SlidingWindow


def dehaze(image_path, airlight=np.array([9.5, 10, 9.5])):
    """ Dehazes an image using color-lines and a given airlight vector. """
    airlight = airlight / np.linalg.norm(airlight)
    img = cv2.imread(image_path) / 255
    transmission_image = np.zeros(img.shape[:2])
    sigma_image = np.zeros(img.shape[:2])
    sliding_window = SlidingWindow(img, scans=5)

    for window in sliding_window:
        patch = window.patch
        color_line = ransac(patch, iterations=3)

        if color_line.valid(airlight):
            transmission_image[window.y, window.x] = color_line.transmission
            sigma_image[window.y, window.x] = color_line.sigma(airlight)

    transmission_image = clip(transmission_image)
    transmission_image = interpolate(transmission_image)
    # transmission_image = mrf_interpolate(transmission_image, sigma_image, img)  # Uncomment for mrf
    img = recover(img, transmission_image, airlight)
    return img


def main():
    dehazed = dehaze(join('images', 'bricks.png'))
    rgb = cv2.cvtColor(dehazed.astype(np.float32), cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()


if __name__ == '__main__':
    main()
