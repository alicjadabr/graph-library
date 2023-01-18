from baseimage import BaseImage, ColorModel
import numpy as np
import image as img
import cv2


class GrayScaleTransform(BaseImage):
    def __init__(self, path: str = None) -> None:
        super().__init__(path)

    def to_gray(self) -> 'Image':
        gray_image = img.Image()
        gray_image.data = np.dot(self.data, [0.299, 0.587, 0.114]).astype('uint8')
        gray_image.color_model = ColorModel.gray
        return gray_image

    def to_gray_cv2(self) -> 'Image':
        img_grayscale = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
        gray_image = img.Image()
        gray_image.data = img_grayscale
        gray_image.color_model = ColorModel.gray
        return gray_image

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None):
        gray_arr = self.to_gray().data
        l0 = gray_arr[:, :].astype('float64')
        l1 = gray_arr[:, :].astype('float64')
        l2 = gray_arr[:, :].astype('float64')

        if all(alpha_beta):
            if alpha_beta[0] > 1.0 > alpha_beta[1] and (alpha_beta[0] + alpha_beta[1] == 2.0):
                l0 *= alpha_beta[0]
                l2 *= alpha_beta[1]
            else:
                return print('Alfa i beta nie speniaja zaleznosci: alfa > 1, beta < 1, alfa + beta = 2')

        else:
            if 20 <= w <= 40:
                l0 += 2 * w
                l1 += w
            else:
                return print('Argument -w- musi byc liczba z przedzialu: <20; 40>')

        sepia_image = img.Image()
        sepia_array: np.ndarray = np.dstack((l0, l1, l2))
        sepia_array[(sepia_array > 255)] = 255
        sepia_image.data = sepia_array.astype('uint8')
        return sepia_image
