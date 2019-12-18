"""
Augmenters that wrap methods from ``bethgelab.imagecorruptions`` package.

See https://github.com/bethgelab/imagecorruptions for the package.

List of augmenters:

    TODO

.. warning::

    The functions in this module will convert ``uint8`` to ``float32``
    or ``float64`` outputs. The wrapped functions from ``imagecorruptions``
    perform this change. This deviates from other augmentation functions
    in imgaug, which always retain the input dtype. The outputs are here
    not converted back to ``uint8`` as that would slightly alter the array
    components and thereby possibly affect evaluations for papers.

"""
from __future__ import print_function, division, absolute_import

import numpy as np

from .. import dtypes as iadt
from .. import random as iarandom

# TODO deal with warnings.simplefilter in imagecorruptions
# TODO docstrings
# TODO add tests for temporary numpy seed
# TODO list known differences
#      fnames are "apply_imgcorrupt_<fname>" instead of "<fname>"
#      added seed parameter
#      expect always numpy inputs
#      always return numpy outputs
#      accept (H,W) and (H,W,1) inputs
# TODO add optional dependency
# TODO add package to test requirements
# TODO add train/test set functions
# TODO add np.uint8() casting, similar to corrupt()
# TODO add augmenters

_MISSING_PACKAGE_ERROR_MSG = (
    "Could not import package `imagecorruptions`. This is an optional "
    "dependency of imgaug and must be installed manually in order "
    "to use augmenters from `imgaug.augmenters.imgcorrupt`. "
    "Use e.g. `pip install imagecorruptions` to install it. See also "
    "https://github.com/bethgelab/imagecorruptions for the repository "
    "of the package."
)


def _normalize_grayscale_inputs(x):
    input_was_grayscale = False
    input_was_grayscale3d = False
    if x.ndim == 2:
        input_was_grayscale = True
        x = x[:, :, np.newaxis]
        x = np.tile(x, (1, 1, 3))
    elif x.ndim == 3 and x.shape[2] == 1:
        input_was_grayscale3d = True
        x = np.tile(x, (1, 1, 3))
    inv_info = (input_was_grayscale, input_was_grayscale3d)
    return x, inv_info


def _invert_normalize_grayscale_inputs(x, inv_info):
    input_was_grayscale, input_was_grayscale3d = inv_info
    if input_was_grayscale3d:
        return x[..., 0:1]
    elif input_was_grayscale:
        return x[..., 0]
    return x


def _call_imgcorrupt_func(fname, seed, convert_to_pil, *args, **kwargs):
    try:
        import imagecorruptions.corruptions as corruptions
    except ImportError:
        raise ImportError(_MISSING_PACKAGE_ERROR_MSG)

    image = args[0]

    iadt.gate_dtypes(
        image,
        allowed=["uint8"],
        disallowed=["bool",
                    "uint16", "uint32", "uint64", "uint128", "uint256",
                    "int8", "int16", "int32", "int64", "int128", "int256",
                    "float16", "float32", "float64", "float96", "float128",
                    "float256"],
        augmenter=None)

    input_shape = image.shape

    if convert_to_pil:
        import PIL.Image
        image = PIL.Image.fromarray(image)

    with iarandom.temporary_numpy_seed(seed):
        image_aug = getattr(corruptions, fname)(image, *args[1:], **kwargs)

    if convert_to_pil:
        image_aug = np.asarray(image_aug)

    if len(input_shape) == 3 and input_shape[2] == 1 and image_aug.ndim == 2:
        image_aug = image_aug[:, :, np.newaxis]

    return image_aug


def apply_imgcorrupt_gaussian_noise(x, severity=1, seed=None):
    return _call_imgcorrupt_func("gaussian_noise", seed, False, x, severity)


def apply_imgcorrupt_shot_noise(x, severity=1, seed=None):
    return _call_imgcorrupt_func("shot_noise", seed, False, x, severity)


def apply_imgcorrupt_impulse_noise(x, severity=1, seed=None):
    return _call_imgcorrupt_func("impulse_noise", seed, False, x, severity)


def apply_imgcorrupt_speckle_noise(x, severity=1, seed=None):
    return _call_imgcorrupt_func("speckle_noise", seed, False, x, severity)


def apply_imgcorrupt_gaussian_blur(x, severity=1, seed=None):
    return _call_imgcorrupt_func("gaussian_blur", seed, False, x, severity)


def apply_imgcorrupt_glass_blur(x, severity=1, seed=None):
    return _call_imgcorrupt_func("glass_blur", seed, False, x, severity)


def apply_imgcorrupt_defocus_blur(x, severity=1, seed=None):
    x, inv_info = _normalize_grayscale_inputs(x)
    result = _call_imgcorrupt_func("defocus_blur", seed, False, x, severity)
    result = _invert_normalize_grayscale_inputs(result, inv_info)
    return result


def apply_imgcorrupt_motion_blur(x, severity=1, seed=None):
    # motion_blur() crashes for both (H,W) and (H,W,1) inputs
    x, inv_info = _normalize_grayscale_inputs(x)
    result = _call_imgcorrupt_func("motion_blur", seed, False, x, severity)
    result = _invert_normalize_grayscale_inputs(result, inv_info)
    return result


def apply_imgcorrupt_zoom_blur(x, severity=1, seed=None):
    # zoom_blur() crashes for (H,W,1) inputs
    x, inv_info = _normalize_grayscale_inputs(x)
    result = _call_imgcorrupt_func("zoom_blur", seed, False, x, severity)
    result = _invert_normalize_grayscale_inputs(result, inv_info)
    return result


def apply_imgcorrupt_fog(x, severity=1, seed=None):
    # fog() crashes for (H,W,1) inputs
    x, inv_info = _normalize_grayscale_inputs(x)
    result = _call_imgcorrupt_func("fog", seed, False, x, severity)
    result = _invert_normalize_grayscale_inputs(result, inv_info)
    return result


def apply_imgcorrupt_frost(x, severity=1, seed=None):
    x, inv_info = _normalize_grayscale_inputs(x)
    result = _call_imgcorrupt_func("frost", seed, False, x, severity)
    result = _invert_normalize_grayscale_inputs(result, inv_info)
    return result


def apply_imgcorrupt_snow(x, severity=1, seed=None):
    # snow() turns (H,W,1) into (H,W,H?)
    x, inv_info = _normalize_grayscale_inputs(x)
    result = _call_imgcorrupt_func("snow", seed, False, x, severity)
    result = _invert_normalize_grayscale_inputs(result, inv_info)
    return result


def apply_imgcorrupt_spatter(x, severity=1, seed=None):
    # spatter() crashes for (H,W,1) inputs
    # for (H,W) inputs the result wihtout normalization differ from a direct
    # spatter() call
    x, inv_info = _normalize_grayscale_inputs(x)
    result = _call_imgcorrupt_func("spatter", seed, True, x, severity)
    result = _invert_normalize_grayscale_inputs(result, inv_info)
    return result


def apply_imgcorrupt_contrast(x, severity=1, seed=None):
    return _call_imgcorrupt_func("contrast", seed, False, x, severity)


def apply_imgcorrupt_brightness(x, severity=1, seed=None):
    return _call_imgcorrupt_func("brightness", seed, False, x, severity)


def apply_imgcorrupt_saturate(x, severity=1, seed=None):
    x, inv_info = _normalize_grayscale_inputs(x)
    result = _call_imgcorrupt_func("saturate", seed, False, x, severity)
    result = _invert_normalize_grayscale_inputs(result, inv_info)
    return result


def apply_imgcorrupt_jpeg_compression(x, severity=1, seed=None):
    # jpeg_compression() requires 3-channel data
    x, inv_info = _normalize_grayscale_inputs(x)
    result = _call_imgcorrupt_func("jpeg_compression", seed, True, x, severity)
    result = _invert_normalize_grayscale_inputs(result, inv_info)
    return result


def apply_imgcorrupt_pixelate(x, severity=1, seed=None):
    return _call_imgcorrupt_func("pixelate", seed, True, x, severity)


def apply_imgcorrupt_elastic_transform(image, severity=1, seed=None):
    # elastic_transform() crashes for (H,W,1) inputs
    image, inv_info = _normalize_grayscale_inputs(image)
    result = _call_imgcorrupt_func("elastic_transform", seed, False, image,
                                   severity)
    result = _invert_normalize_grayscale_inputs(result, inv_info)
    return result

