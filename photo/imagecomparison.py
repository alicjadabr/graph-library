from enum import Enum
from grayscale import GrayScaleTransform
from histogram import Histogram
from math import sqrt


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1


class ImageComparison(GrayScaleTransform):
    def histogram(self) -> Histogram:
        # metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        image_histogram = Histogram(self.data)
        return image_histogram

    def compare_to(self, other: 'Image', method: ImageDiffMethod) -> float:
        x_hist = Histogram(self.to_gray().data)
        y_hist = Histogram(other.to_gray().data)
        sum: float = 0.0
        for x in range(x_hist.values.shape[0]):
            for y in range(x_hist.values.shape[1]):
                diff: float = float(x_hist.values[x, y]) - float(y_hist.values[x, y])
                sum += diff * diff

        mse: float = int(sum) / float(x_hist.values.shape[0] * x_hist.values.shape[1])

        if method == ImageDiffMethod.mse:
            return mse
        elif method == ImageDiffMethod.rmse:
            rmse: float = sqrt(mse)
            return rmse
