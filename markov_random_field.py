import autograd.numpy as np
from autograd import elementwise_grad as grad
from autograd.scipy.signal import convolve
from scipy.ndimage.morphology import grey_dilation as dialate

import constants


def mrf_interpolate(transmission_image, sigma_image, img):
    """ Interpolate the transmission image with a Markov random field. """
    width = constants.patch_size
    transmission_image[transmission_image < 0.3] = 0.3
    transmission_image = dialate(transmission_image, size=(width, width))

    sigma_image = dialate(sigma_image, size=(width, width))
    sigma_image[sigma_image == 0] = constants.sigma_default
    interpol_image = np.full(transmission_image.shape, 0.6)
    interpol_image[transmission_image == 0] = 0
    interpol_image = interpol_image + transmission_image

    grad_data_term = grad(data_term, 0)
    grad_regularization_term = grad(regularization_term, 0)

    for _ in range(constants.epochs):
        data_error_grad = grad_data_term(transmission_image, interpol_image, sigma_image)
        data_error_grad[transmission_image == 0] = 0
        data_error = data_error_grad.sum()

        regularization_error_grad = grad_regularization_term(interpol_image, img)
        regularization_error = regularization_error_grad.sum()
        regularization_error_grad[regularization_error_grad > 2] = 2
        regularization_error_grad[regularization_error_grad < -2] = -2

        interpol_image += constants.learning_rate * regularization_error_grad
        interpol_image += constants.learning_rate * data_error_grad

        print('reg error: {}'.format(-regularization_error))
        print('data error: {}'.format(data_error))

    interpol_image[interpol_image > 1] = 1
    interpol_image[interpol_image < 0.3] = 0.3
    return interpol_image


def data_term(transmission_image, interpol_image, sigma_image):
    """ The data term of the error.

        This term is responsible for enforcing the estimated transmissions to
        appear in the interpolated image.
    """
    diff = (interpol_image - transmission_image)
    diff_squared = diff ** 2
    sigma_squared = sigma_image ** 2
    error = diff_squared / sigma_squared
    return error


def regularization_term(interpol_image, image):
    """ The regularization term of the error.

        This term is responsible for smoothing the image according to
        the pixel differences present in the rgb image.
    """
    filter = np.full((51, 51), -1)
    filter[24, 24] = 2600

    b, g, r = np.split(image, 3, axis=2)

    b_diff = convolve(filter, np.squeeze(b))
    g_diff = convolve(filter, np.squeeze(g))
    r_diff = convolve(filter, np.squeeze(r))

    image_diff = np.stack((b_diff, g_diff, r_diff), axis=-1)
    image_diff = np.linalg.norm(image_diff, axis=2)
    interpol_diff = convolve(interpol_image, filter)
    error = (interpol_diff / image_diff) ** 2
    return -error
