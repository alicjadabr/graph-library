from imagecomparison import ImageComparison
import cv2
import numpy as np
import image as img
import histogram as hist


class ImageAligning(ImageComparison):
    def __init__(self, path: str = None) -> None:
        super().__init__(path)

    def align_image(self, tail_elimination: bool = False) -> 'Image':

        def count_layer(layer: np.ndarray):
            if tail_elimination is False:
                min = np.min(layer)
                max = np.max(layer)
            elif tail_elimination is True:
                cumu_hist = hist.Histogram(layer).to_cumulated().values
                min, max = 0, 0
                for x in cumu_hist:
                    if x < cumu_hist[-1] * 0.05:
                        min += 1
                for y in cumu_hist:
                    if y < cumu_hist[-1] * 0.95:
                        max += 1

            layer = layer.astype('float64')
            layer[:, :] = (layer[:, :] - min) * (255 / (max - min))
            layer[(layer > 255)] = 255
            layer[(layer < 0)] = 0
            return layer

        if self.data.ndim == 3:
            img_arr = np.dstack((count_layer(self.get_layer(0).data),
                                 count_layer(self.get_layer(1).data), count_layer(self.get_layer(2).data)))
        elif self.data.ndim == 2:
            img_arr = count_layer(self.data)

        new_image = img.Image()
        new_image.data = img_arr.astype('uint16')
        return new_image

    def clahe(self) -> 'Image':
        if self.data.ndim == 2:
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(4, 4)
            )
            equalized_image = clahe.apply(self.data)

        # Korekcja histogramu w obrazach kolorowych
        elif self.data.ndim == 3:
            img_lab = cv2.cvtColor(self.data, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_lab[..., 0] = clahe.apply(img_lab[..., 0])
            equalized_image = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

        new_image = img.Image()
        new_image.data = equalized_image
        return new_image
