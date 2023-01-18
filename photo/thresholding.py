from imagefiltration import ImageFiltration
import numpy as np
import cv2
import image as img


class Thresholding(ImageFiltration):
    def threshold(self, value: int = 125) -> 'Image':
        result_array: np.ndarray = self.data.copy()
        result_array[(result_array < value)] = 0
        result_array[(result_array >= value)] = 255

        image = img.Image()
        image.data = result_array.astype('uint8')
        return image

    def otsu(self) -> 'Image':
        # Implementacja progowania metodą Otsu
        if self.data.ndim == 3:
            self = self.to_gray()

        _, thresh_otsu = cv2.threshold(
        self.data,
        thresh=0,
        maxval=255,
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        image = img.Image()
        image.data = thresh_otsu
        return image

    def adaptiveThreshold(self, max_value: int = 255, method: str = 'mean', type: str = 'binary', block_size = 13, c = 8) -> 'Image':
        # Implementacja progowania metodą adaptacyjną
        if self.data.ndim == 3:
            self = self.to_gray()

        if method == 'mean':
            method = cv2.ADAPTIVE_THRESH_MEAN_C
        elif method == 'gauss':
            method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            print('Nieprawidłowa metoda, podaj =mean= lub =gauss=')

        if type == 'binary':
            type = cv2.THRESH_BINARY
        elif type == 'binary_inv':
            type = cv2.THRESH_BINARY_INV
        else:
            print('Nieprawidłowa metoda, podaj =binary= lub =binary_inv=')

        th_adaptive = cv2.adaptiveThreshold(
        self.data,
        maxValue=max_value,
        adaptiveMethod=method,
        thresholdType=type,
        blockSize=block_size,
        C=c
        )
        image = img.Image()
        image.data = th_adaptive
        return image
