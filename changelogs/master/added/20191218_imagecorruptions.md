# Added Wrappers for `imagecorruptions` Package

Added wrappers around the functions from package
[bethgelab/imagecorruptions](https://github.com/bethgelab/imagecorruptions).
The functions in that package were used in some recent papers and are added
here for convenience.
The wrappers produce arrays containing values identical to the output
arrays from the corresponding `imagecorruptions` functions (verified via
unittests).
They do however always output numpy arrays (instead of sometimes PIL images)
and always expect numpy arrays as inputs (instead of expecting PIL images and
also sometimes accepting numpy arrays).
The interfaces of the wrapper functions are identical to the
`imagecorruptions` functions, with the only difference of also supporting
`seed` parameters.

Note that to ensure identical outputs the implemented functions always wrap
functions from `imagecorruptions`, even when faster and more robust
alternatives exist in `imgaug`.

* Added module `imgaug.augmenters.imgcorrupt`.
* Added functions to module `imgaug.augmenters.imgcorrupt`:
    * `apply_imgcorrupt_gaussian_noise()`
    * `apply_imgcorrupt_shot_noise()`
    * `apply_imgcorrupt_impulse_noise()`
    * `apply_imgcorrupt_speckle_noise()`
    * `apply_imgcorrupt_gaussian_blur()`
    * `apply_imgcorrupt_glass_blur()`
    * `apply_imgcorrupt_defocus_blur()`
    * `apply_imgcorrupt_motion_blur()`
    * `apply_imgcorrupt_zoom_blur()`
    * `apply_imgcorrupt_fog()`
    * `apply_imgcorrupt_snow()`
    * `apply_imgcorrupt_spatter()`
    * `apply_imgcorrupt_contrast()`
    * `apply_imgcorrupt_brightness()`
    * `apply_imgcorrupt_saturate()`
    * `apply_imgcorrupt_jpeg_compression()`
    * `apply_imgcorrupt_pixelate()`
    * `apply_imgcorrupt_elastic_transform()`
* Added context `imgaug.random.temporary_numpy_seed()`.
