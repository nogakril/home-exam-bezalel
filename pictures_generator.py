import os

import numpy as np
from scipy.ndimage import convolve
import scipy.signal as signal
from imageio import imread, imwrite
from skimage.color import rgb2gray

GRAYSCALE = 1
RGB = 2
MAX_COLOR_VAL = 255
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


def stretch(pyr):
    den = np.max(pyr) - np.min(pyr)
    nom = pyr - np.min(pyr)
    if den > 0:
        return nom / den
    return nom


def resize(pyr, height):
    return np.pad(pyr, ((0, height - pyr.shape[0]), (0, 0)))


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    if levels == 1:
        return stretch(pyr[0])
    stretch_vec = np.vectorize(stretch, otypes=[np.ndarray])
    resize_vec = np.vectorize(resize, otypes=[np.ndarray])
    stretched_pyr = stretch_vec(pyr[:min(levels, len(pyr))])
    stretched_and_resized = resize_vec(stretched_pyr, pyr[0].shape[0])
    return np.hstack(stretched_and_resized)


def calc_max_level(size):
    k = 0
    while size % 2 == 0:
        size /= 2
        k += 1
    return k


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    blurred_im = convolve(convolve(im, blur_filter), blur_filter.T)
    reduced_im = blurred_im[::2, ::2]
    return reduced_im


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    padded_im = np.zeros((2 * im.shape[0], 2 * im.shape[1]))
    padded_im[0::2, 0::2] = im
    blurred_im = convolve(convolve(padded_im, blur_filter), blur_filter.T)
    return blurred_im


def get_gaussian_filter(filter_size):
    gaussian_filter = np.array([[1, 1]], dtype=np.uint64)
    while gaussian_filter.size < filter_size:
        gaussian_filter = signal.convolve(np.array([[1, 1]], dtype=np.uint64), gaussian_filter)
    return gaussian_filter / np.sum(gaussian_filter)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    filter_vec = get_gaussian_filter(filter_size)
    pyr = [im]
    curr_level_im = im
    for i in range(max_levels - 1):
        curr_level_im = reduce(curr_level_im, filter_vec)
        if min(curr_level_im.shape) <= 16:
            break
        pyr.append(curr_level_im)
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    for i in range(len(gaussian_pyr) - 1):
        pyr.append(gaussian_pyr[i] - expand(gaussian_pyr[i + 1], filter_vec * 2))
    pyr.append(gaussian_pyr[-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    im = lpyr[-1] * coeff[-1]
    for i in range(len(lpyr) - 2, -1, -1):
        im = (coeff[i] * lpyr[i]) + expand(im, filter_vec * 2)
    return im


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    L1, filter_vec_1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filter_vec_2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    Gm, filter_vec_m = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    L_out = []
    for i in range(len(L1)):
        L_out.append((Gm[i] * L1[i]) + ((1 - Gm[i]) * L2[i]))
    im_blend = laplacian_to_image(L_out, filter_vec_1, list(np.ones(len(L_out))))
    return np.clip(im_blend, 0, 1)


def blender_helper(im1, im2, mask, output_path):
    max_level = min([calc_max_level(im1.shape[0]), calc_max_level(im1.shape[1])])
    blended = np.zeros_like(im1)
    blended[:, :, 0] = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_level, 45, 45)
    blended[:, :, 1] = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, max_level, 45, 45)
    blended[:, :, 2] = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, max_level, 45, 45)

    imwrite(output_path, blended)
    return im1, im2, mask.astype(np.bool_), blended


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = imread(filename).astype(np.float64)
    if representation == GRAYSCALE:
        image = rgb2gray(image)
    return image / MAX_COLOR_VAL


def blend_pictures(picture1_path, picture2_path, mask_path, output_path, scale):
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath(picture1_path),
                     RGB)
    im2 = read_image(relpath(picture2_path),
                     RGB)
    mask = np.round(read_image(relpath(mask_path), GRAYSCALE))
    mask[mask == 0] = scale
    return blender_helper(im1, im2, mask, output_path)


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)
