import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ransac import ransac
from window import SlidingWindow


def dehaze(image_path):
    """ Dehazes an image using color-lines and a given airlight vector. """

    img = cv2.imread(image_path) / 255
    transmission_image = np.zeros(img.shape)
    sliding_window = SlidingWindow(img)
    airlight = np.array([0.9, 0.97, 0.988])

    for window in sliding_window:
        patch = window.patch
        color_line = ransac(patch, iterations=3)

        if color_line.valid(airlight=airlight):
            transmission_image[window.y, window.x] = color_line.transmission

    plt.imshow(transmission_image)
    plt.show()
    print(transmission_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='Image to dehaze', type=str)
    args = parser.parse_args()
    image_path = args.image
    dehaze('bricks.png')


if __name__ == '__main__':
    main()
