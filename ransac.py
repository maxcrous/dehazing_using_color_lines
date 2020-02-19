import numpy as np

from color_line import ColorLine


def choice(array, size, replace):
    """ Returns a sample of a certain size for a numpy array. """
    indices = np.random.choice(len(array), size=size, replace=replace)
    sample = array[indices]
    return sample


def ransac(patch, iterations=30, threshold=0.02):
    """Returns the color-line of a patch in vector form.

    Iterates over pairs of 2 pixels. The line between each pixel pair in
    RGB space is considered a line hypothesis. A certain number (`iterations`)
    of these hypotheses are considered. The line that fits the most pixels is
    returned in vector form.

    Args:
        patch (3D numpy array): A square patch of an image.
        iterations (int): The number of color-lines that are tested.
        threshold (float): The distance threshold between the hypothesis line
        and a point for the line to be considered a good fit for the point.
    Returns:
        color_line (ColorLine): An object holding all information related to
        the color line.
    """
    best_support = -np.inf
    direction = None
    point = None
    patch_flat = np.reshape(patch, (-1, 3))

    for _ in range(iterations):
        v1, v2 = choice(patch_flat, size=2, replace='False')
        d1 = v2 - v1
        support = 0

        for pixel in patch_flat:
            d2 = pixel - v1
            dist = distance(d1, d2)

            if dist < threshold:
                support += 1

        if support > best_support:
            point = v1
            direction = d1
            best_support = support

    support_matrix = create_support_matrix(patch, point, direction, threshold)
    color_line = ColorLine(point, direction, patch, support_matrix)
    return color_line


def distance(d1, d2):
    """ Returns the distance metric for two direction vectors.

    The distance metric determines whether a color-line fits
    a pixel. Fattal defines this distance metric as the projection
    of a pixels direction vector onto the plane that is perpendicular
    to the color-lines direction vector.

    Args:
        d1: A color-line direction vector.
        d2: A direction vector.
    Returns:
        dist: The distance metric.
    """
    projection_onto_plane = d2 - projection(d1, d2)
    dist = np.linalg.norm(projection_onto_plane)

    return dist


def projection(v1, v2):
    """ Returns the projection of v2 onto v1.

    Args:
        v1: A vector that is projected onto.
        v2: A vector that is projected.
    Returns:
        result: A vector projection.
    """
    if not np.array_equal(v1, np.zeros(3)):
        result = (np.dot(v1, v2) / np.dot(v1, v1)) * v1
    else:
        result = np.zeros(3)
    return result


def create_support_matrix(patch, point, direction, threshold):
    """ Returns a matrix encoding the support for the color-line

    Given a threshold, each pixel in a patch either does or doesn't
    fit the color-line. This matrix has the same size as a patch and
    contains `True` if that patch pixel supports the color-line.

    Args:
        patch: A square patch of an image.
        point: A point on the color line.
        direction: The color-line direction vector.
        threshold: The maximum distance a point can have from the
        color-line in order to support the line.
    Returns:
        support_matrix: A boolean matrix encoding pixel support.
    """
    height, width, _ = patch.shape
    support_matrix = np.full((height, width), False, dtype=bool)

    for idy, row in enumerate(patch):
        for idx, pixel in enumerate(row):
            d2 = pixel - point
            dist = distance(direction, d2)
            if dist < threshold:
                support_matrix[idy][idx] = True

    return support_matrix
