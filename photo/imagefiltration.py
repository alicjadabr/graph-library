from imagealigning import ImageAligning
from typing import Optional
import image as img
import numpy as np
import cv2


class ImageFiltration(ImageAligning):
    def conv_2d(self, kernel: np.ndarray, prefix: Optional[float] = 1) -> 'Image':
        size_kernel = kernel.shape[0]
        kernel = (kernel * prefix).astype('float64')

        def count_total(matrix: np.ndarray):
            sum: float = 0
            for w in range(size_kernel):
                for k in range(size_kernel):
                    sum += kernel[w, k] * matrix[w, k]
            return sum

        def count_layer(layer: np.ndarray) -> np.ndarray:
            if size_kernel == 3:
                limit = 2
            elif size_kernel == 5:
                limit = 3

            filter_arr: np.ndarray = layer.copy().astype('float64')
            for x in range(size_kernel-limit, layer.shape[0]-limit):
                for y in range(size_kernel-limit, layer.shape[1]-limit):
                    filter_arr[x, y] = count_total(layer[x-limit+1:x+limit, y-limit+1:y+limit].astype('float64'))
            return filter_arr

        if self.data.ndim == 2:
            filtered_array = count_layer(self.data)
        elif self.data.ndim == 3:
            filtered_array = np.dstack((count_layer(self.data[:,:,0]),count_layer(self.data[:,:,1]),count_layer(self.data[:,:,2])))

        filtered_array[(filtered_array > 255)] = 255
        filtered_array[(filtered_array < 0)] = 0
        filtered_image = img.Image()
        filtered_image.data = filtered_array.astype('uint8')
        return filtered_image

    def edge_detection(self) ->'Image':
        omega0 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        omega45 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
        omega90 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        omega135 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])

        all_edges = img.Image()
        all_edges.data = (self.conv_2d(omega0).data + self.conv_2d(omega45).data
        + self.conv_2d(omega90).data + self.conv_2d(omega135).data).astype('uint16')

        return all_edges

    def canny(self, th1: int = 14, th2: int = 44, kernel_size: int = 3) -> 'Image':
        #Detekcja krawędzi metodą Canny'ego
        canny_edges = cv2.Canny(
        self.data,
        threshold1=th1,  # prog histerezy 1
        threshold2=th2,  # prog histerezy 2
        apertureSize=kernel_size  # wielkoscc filtra sobela
        )

        image = img.Image()
        image.data = canny_edges
        return image