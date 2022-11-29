# CV Operator Optimization Samples by using Intel(r) oneAPI Libraries (IPP, oneTBB, oneMKL)

| Operator name                             | Supported Intel(r) Architecture(s) | Description
|:---                                       |:---                                |:---
| Sobel gradient magnitude(sum)             | CPU  | The calculations of the gradients magnitude is measured by the sum of absolute values of the gradient in both directions.
| Sobel gradient magnitude(hypot)           | CPU  | The calculations of the gradients magnitude is measured by the hypotenuse of the triangle of the gradient in both directions.
| Gaussian blur(3x3,5x5,21x21)              | CPU  | The calculations of the gaussian blur operation, the kernel size is 3x3, 5x5, 21x21.
| Image difference                        | CPU  | The sample code for calculating the difference of two images in IPP and OpenCV
| Morphology Open		      | CPU  | The sample code of image Morphology Open	operation by IPP and OpenCV, the kernel size is 3x3, 5x5.
| Warp affine				    | CPU  | The sample code of warp affine by IPP+openMP and OpenCV
| Image Flip			      | CPU  | The sample code of image flip by IPP and OpenCV
| Histogram			      | CPU  | The sample code of calculating the histogram of image by IPP and OpenCV
| Mean Filter			      | CPU  | The sample code of image smoothing by IPP and OpenCV using kernel 11x11

## Trouble shooting
Because of the [limit of windows path length](https://stackoverflow.com/questions/17807281/visual-studio-pathtoolongexception-even-when-the-path-length-is-less-than-260) when using Visual Studio (max path length allowed while creating a project is 260 characters and 248 characters for directory.). If you cannot open the project directly with Visual studio inside the CVOI package, please copy the project folder out to your own workspace and rerun the project following its guide inside.
