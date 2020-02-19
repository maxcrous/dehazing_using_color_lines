import math
import numpy as np
import vg

import thresholds

np.seterr(divide='ignore', invalid='ignore')


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
        """ Change sign of the direction vector when it's negative. """

        for elem in self.direction:
            if elem < 0:
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
        """ Test whether enough points support a color-line. """
        total_votes = self.patch.size
        threshold = thresolds.support * total_votes

        if self.support_matrix.sum() < threshold:
            return False
        else:
            return True

    def positive_reflectance(self):
        """ Ensure the color-line doesn't have mixed signs in its direction vector. """
        for elem in self.direction:
            if elem < 0:
                return False
        return True

    def large_intersection_angle(self, airlight):
        """ Ensure the angle between the color-line orientation and
            atmospheric light vector is large enough.
        """
        angle = vg.angle(airlight, self.direction)

        return angle > thresholds.angle

    def unimodality(self):
        """ Ensure the support points projected onto the color-line
            follow a unimodal distribution.
        """
        a, b = self.normalize_coefficients()

        total_score = 0

        for idy, row in enumerate(self.patch):
            for idx, pixel in enumerate(row):
                if self.support_matrix[idy][idx]:
                    direction = pixel - self.point
                    product = np.dot(direction, self.direction)
                    score = math.cos(a * (product + b))
                    total_score += score

        total_score = total_score / np.sum(self.support_matrix)
        total_score = abs(total_score)
        return total_score < thresholds.unimodal

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
        """ Ensure the airlight and color-line (almost) intersect.

            Algorithm taken from http://geomalgorithms.com/a07-_distance.html,
            See https://www.youtube.com/watch?v=HC5YikQxwZA for algebraic solution.
        """
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
        return length < thresholds.intersection

    def valid_transmission(self):
        """ Ensure the transmission falls within a valid range. """
        return 0 < self.transmission < 1

    def sufficient_shading_variability(self):
        """ Ensure there is sufficient variability in the shading. """
        samples = []
        for idy, row in enumerate(self.patch):
            for idx, pixel in enumerate(row):
                if self.support_matrix[idy][idx]:
                    direction = pixel - self.point
                    samples.append(direction)

        samples = np.array(samples)
        variance = np.var(samples)
        score = np.sqrt(variance) / self.transmission

        return score > thresholds.shading

    def calculate_transmission(self, airlight):
        """ Determine the transmission, given the color-line and airlight.

            Algorithm taken from appendix in Fattal's paper.
        """
        d_unit = self.direction / np.linalg.norm(self.direction)
        a_unit = airlight / np.linalg.norm(airlight)

        ad = np.dot(a_unit, d_unit)
        dv = np.dot(d_unit, self.point)
        av = np.dot(a_unit, self.point)

        s = (np.dot(-dv, ad) + av) / (1 - np.dot(ad, ad))
        self.transmission = 1 - s
