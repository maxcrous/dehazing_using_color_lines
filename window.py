
class SlidingWindow:
    """ An iterator class that acts as a window function over an image.

    Estimating the color-line and transmission at every possible image
    window is costly and redundant due to overlap. This sliding window
    class is used to iterate over a subset of all image windows. This
    is achieved by scanning the image in non-overlapping patches. To
    avoid large unresolved areas, multiple passes are made with
    slightly different window offsets.

     Example use:

        sliding_window = SlidingWindow(image)
        for window in sliding_window:
            foo(...)
     Args:
            image (3D numpy array): The image the window slides over.
            patch_size (int): The size of individual patches
            scans (int): The number of times the non-overlapping sliding
            window passes over the entire image with a small offset.
    """

    def __init__(self, image, patch_size=7, scans=4):
        self.image = image
        self.patch_size = patch_size
        self.scans = scans

    def __iter__(self):
        height, width, _ = self.image.shape
        patch_size = self.patch_size
        hop_size = self.patch_size
        image = self.image

        for offset in range(self.scans):
            x_start = 0 + offset * (patch_size - 1 // 2)
            y_start = 0 + offset * (patch_size - 1 // 2)
            x_end = width - patch_size + 1
            y_end = height - patch_size + 1

            for y in range(y_start, y_end, hop_size):
                for x in range(x_start, x_end, hop_size):
                    patch = image[y:y + patch_size,
                                  x:x + patch_size, :]

                    window = Window(patch,
                                    y=y + patch_size - 1 // 2,
                                    x=x + patch_size - 1 // 2)
                    yield(window)


class Window:
    """ A class containing an image pixel patch and its coordinates. """

    def __init__(self, patch, x, y):
        self.patch = patch
        self.x = x
        self.y = y
