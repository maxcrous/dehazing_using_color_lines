import numpy as np
import math
import vg


class ColorLine:
    """ A color-line represented in vector form.

     Args:
            point (3D numpy array): A vector of a point on the line.
            direction (3D numpy array): The direction vector of the line.
    """

    def __init__(self, point, direction, patch, support_matrix):
        self.point = point
        self.direction = direction
        self.patch = patch
        self.transmission = None
        self.support_matrix = support_matrix
        self.direction_sign()

    def direction_sign(self):
        """ Change sign of direction vector when it's negative. """
        if self.direction[0] < 0:
            self.direction = -self.direction

    def valid(self, airlight):
        """ Returns True when a color-line passes all quality tests. """

        self.calculate_transmission(airlight)
        passed_all_tests = (self.significant_line_support
                            and self.positive_reflectance()
                            and self.large_intersection_angle(airlight)
                            and self.unimodality()
                            and self.close_intersection(airlight)
                            and self.valid_transmission()
                            and self.sufficient_shading_variability())
        return passed_all_tests

    def significant_line_support(self):
        """
        A small number of supporting pixels implies that either the line fails
        to represent the patch pixels or that most of its pixels do not obey
        Eq. (4) as its underlying assumptions do not hold. Therefore, we discard
        lines with less than 40% pixel support in the patch. If the line passes
        this test, we redefine the set of patch pixels Ω to be the subset of
        pixels that support it and do not consider the rest of the pixels in
        the following tests.
        """
        total_votes = self.patch.size
        threshold = 0.4 * total_votes

        if self.support_matrix.sum() < threshold:
            return False
        else:
            return True

    def positive_reflectance(self):
        """
        The color-line orientation D, as discussed in Section 3.3, corresponds to
        the surface reflectance vector R in Eq. (4). Therefore, we discard lines in
        which negative values are found in its orientation vector D. More precisely,
        since we obtain D up to an arbitrary factor, we identify this inconsistency
        when D’s show mixed signs.
        """
        for elem in self.direction:
            if elem < 0:
                return False
        return True

    def large_intersection_angle(self, airlight):
        """
        The operation of computing the inter- section of two lines, as we do in Eq. (5),
        becomes more sensitive to noise as their orientation gets closer. At the Appendix
        we show that the error in the estimated transmission grows like O(θ−1), where θ is
        the angle between the line orientation D and atmospheric light vector A. Thus, we
        discard lines with θ < 15◦ and weigh the confidence of the estimated transmission
        accordingly when interpolating these values to a complete transmission map (explained below).
        Figure 5 shows an example of patches with small and large inter- section angles.
        """
        threshold = math.pi / 12
        angle = vg.angle(airlight, self.direction)

        return angle > threshold

    def unimodality(self):
        """ """
        a, b = self.normalize_coefficients()

        total_score = 0
        threshold = 0.07

        for idy, row in enumerate(self.patch):
            for idx, pixel in enumerate(row):
                if self.support_matrix[idy][idx]:
                    direction = pixel - self.point
                    product = np.dot(direction, self.direction)
                    score = math.cos(a * (product + b))
                    total_score += score

        total_score = total_score / np.sum(self.support_matrix)
        total_score = abs(total_score)
        return total_score < threshold

    def normalize_coefficients(self):
        """ Returns the variables a and b needed for normalizing
            the distribution when checking for unimodality.
        """
        max_product = -np.inf
        min_product = np.inf

        for idy, row in enumerate(self.patch):
            for idx, pixel in enumerate(row):
                if self.support_matrix[idy][idx]:
                    direction = pixel - self.point
                    product = np.dot(direction, self.direction)

                    if product < min_product:
                        min_product = product
                    if product > max_product:
                        max_product = product

        a = math.pi / (max_product - min_product)
        b = -min_product
        return a, b

    def close_intersection(self, airlight):
        """ Taken from http://geomalgorithms.com/a07-_distance.html and
            https://www.youtube.com/watch?v=HC5YikQxwZA
        """
        threshold = 0.05
        v = airlight
        u = self.direction
        w = v
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w)
        e = np.dot(v, w)
        dd = a * c - b * b
        sc = (b * e - c * d) / dd
        tc = (a * e - b * d) / dd
        dp = w + (sc * u) - (tc * v)
        length = np.linalg.norm(dp)
        return length < threshold

    def valid_transmission(self):
        return 0 < self.transmission < 1

    def sufficient_shading_variability(self):
        threshold = 0.02

        samples = []
        for idy, row in enumerate(self.patch):
            for idx, pixel in enumerate(row):
                if self.support_matrix[idy][idx]:
                    direction = pixel - self.point
                    samples.append(direction)

        samples = np.array(samples)
        variance = np.var(samples)
        score = np.sqrt(variance) / self.transmission

        return score > threshold

    def calculate_transmission(self, airlight):
        d_unit = self.direction / np.linalg.norm(self.direction)
        a_unit = airlight / np.linalg.norm(airlight)

        ad = np.dot(a_unit, d_unit)
        dv = np.dot(d_unit, self.point)
        av = np.dot(a_unit, self.point)

        s = (np.dot(-dv, ad) + av) / (1 - np.dot(ad, ad))
        self.transmission = 1 - s
