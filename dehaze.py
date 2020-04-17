import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ransac import ransac
from recover import recover
from window import SlidingWindow


def dehaze(image_path, airlight=np.array([0.9, 0.97, 0.988])):
    """ Dehazes an image using color-lines. """
    img = cv2.imread(image_path) / 255
    transmission_image = np.zeros(img.shape[:2])
    sliding_window = SlidingWindow(img)

    for window in sliding_window:
        patch = window.patch
        color_line = ransac(patch, iterations=3)

        if color_line.valid(airlight=airlight):
            transmission_image[window.y, window.x] = color_line.transmission

    # transmission_image = interpolate(transmission_image)
    img = recover(img, transmission_image, airlight)
    return img


def main():
    dehazed = dehaze('bricks.png')
    plt.imshow(dehazed)
    plt.show()


if __name__ == '__main__':
    main()
