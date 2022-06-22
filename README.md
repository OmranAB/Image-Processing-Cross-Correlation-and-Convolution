# Image Processing Implemention of Cross-Correlation and Convolution Operations.
The operations are implemented for the three modes 'Full', 'Valid', 'Same'. To assist in padding.
- **Full:** The output is the full discrete linear cross-correlation of the input.
- **Valid:** The output consists only of those elements that do not rely on the zero-padding.
- **Same:** The output is the same size as the input image, centered with respect to the ‘full’ output.

## Cross Correlation:
Apply a filter f or kernel on an image to produce a new image h.


## Convolution:
Convolution is the same as correlation with a 180° rotated filter kernel.


### Assumptions: Image and filter are 2D numpy arrays. You can assume that filter will always be smaller than image. You can
assume that the filter will be odd sized shape (e.g. shape=(3, 5)); this makes computing padding easier.
