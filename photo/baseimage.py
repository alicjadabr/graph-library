import numpy as np
import image as img
from enum import Enum
from matplotlib.image import imread
from matplotlib.image import imsave
from matplotlib.pyplot import imshow


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # obraz 2d


class BaseImage:
    data: np.ndarray
    color_model: ColorModel
    path: str

    def __init__(self, path: str = None) -> None:
        self.color_model = ColorModel.rgb
        if path is None:
            return
        else:
            self.data = imread(path)
            self.path = path

    def save_img(self, path: str) -> None:
        imsave(path, self.data)

    def show_img(self) -> None:
        if self.data.ndim == 3:
            imshow(self.data)
        elif self.data.ndim == 2:
            imshow(self.data, cmap='gray')
        else:
            print("Metoda wyświetla obrazy wyłącznie w modelu barw RGB lub w odcieniach szarości")

    def get_layer(self, layer_id: int):
        if layer_id != 0 and layer_id != 1 and layer_id != 2:
            return print('Incorrect argument. Type: 0 - red layer, 1 - green layer, 2 - blue layer')

        r_layer, g_layer, b_layer = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
        if layer_id == 0:
            selected_layer = r_layer
        elif layer_id == 1:
            selected_layer = g_layer
        elif layer_id == 2:
            selected_layer = b_layer

        new_image = img.Image()
        new_image.data = selected_layer
        new_image.color_model = ColorModel.gray
        return new_image

    def looping_conversion(self, color_model: str):
        # zastosowałam wzór z: https://www.rapidtables.com/convert/color/rgb-to-hsv.html
        cmodel_array: np.ndarray = np.empty(self.data.shape)
        rows, cols, channels = self.data.shape
        for x in range(rows):
            for y in range(cols):
                r, g, b = self.data[x, y][0] / 255, self.data[x, y][1] / 255, self.data[x, y][2] / 255
                max_rgb = max(r, g, b)
                min_rgb = min(r, g, b)
                delta = max_rgb - min_rgb

                if delta == 0:
                    h = 0
                if r == max_rgb:
                    h = (((g - b) / delta) % 6) * 60
                elif g == max_rgb:
                    h = ((b - r) / delta + 2) * 60
                elif b == max_rgb:
                    h = ((r - g) / delta + 4) * 60

                if color_model == 'hsi':
                    last_one = (r * 255 + g * 255 + b * 255) / 3
                    if max_rgb == 0:
                        s = 0
                    else:
                        s = delta / max_rgb * 100
                elif color_model == 'hsv':
                    last_one = max_rgb * 100
                    if max_rgb == 0:
                        s = 0
                    else:
                        s = delta / max_rgb * 100
                elif color_model == 'hsl':
                    last_one = 0.5 * (max_rgb + min_rgb) * 100
                    if delta == 0:
                        s = 0
                    else:
                        s = delta / (1 - abs(2 * (last_one / 100) - 1)) * 100

                cmodel_array[x, y][0] = h
                cmodel_array[x, y][1] = s
                cmodel_array[x, y][2] = last_one

        image = img.Image()
        image.data = cmodel_array.astype('uint16')
        return image

    def to_hsv(self) -> 'Image':
        hsv_image = self.looping_conversion('hsv')
        hsv_image.color_model = ColorModel.hsv
        return hsv_image

    def to_hsl(self) -> 'Image':
        hsl_image = self.looping_conversion('hsl')
        hsl_image.color_model = ColorModel.hsl
        return hsl_image

    def to_hsi(self) -> 'Image':
        hsi_image = self.looping_conversion('hsi')
        hsi_image.color_model = ColorModel.hsi
        return hsi_image

    def to_rgb(self):
        rgb_array: np.ndarray = np.empty(self.data.shape)
        self.data = self.data.astype('float16')
        rows, cols, channels = self.data.shape
        for x in range(rows):
            for y in range(cols):
                # if self.color_model == ColorModel.hsi:
                #     h, s, i = self.data[x,y][0], self.data[x,y][1], self.data[x,y][2]
                #     if 0 <= h < 2*pi/3:
                #         rgb_array[x,y][2] = i * (1 - s)
                #         rgb_array[x,y][0] = i * (1 + s * cos(h / cos(pi/3 - h)))
                #         rgb_array[x,y][1] = 3 * i - (rgb_array[x,y][0] + rgb_array[x,y][2])
                #     elif 2*pi/3 <= h < 4*pi/3:
                #         rgb_array[x,y][0] = i * (1 - s)
                #         rgb_array[x,y][1] = i * (1 + s * cos(h - 2*pi/3) / cos(pi - h))
                #         rgb_array[x,y][2] = 3 * i - (rgb_array[x,y][0] + rgb_array[x,y][1])
                #     elif 4*pi/3 <= h < 2*pi:
                #         rgb_array[x,y][1] = i * (1 - s)
                #         rgb_array[x,y][2] = i * (1 + s * cos(h - 4*pi/3) / cos(5*pi/3 - h))
                #         rgb_array[x,y][0] = 3 * i - (rgb_array[x,y][1] + rgb_array[x,y][2])

                if self.color_model == ColorModel.hsv or self.color_model == ColorModel.hsl:
                    h, s, last_one = self.data[x, y][0], self.data[x, y][1] / 100, self.data[x, y][2] / 100
                    if self.color_model == ColorModel.hsv:
                        c = last_one * s
                        z = c * (1 - abs((h / 60) % 2 - 1))
                        m = last_one - c
                    elif self.color_model == ColorModel.hsl:
                        c = s * (1 - abs(2 * last_one - 1))
                        m = last_one - 0.5 * c
                        z = c * (1 - abs((h / 60) % 2 - 1))
                    f1, f2, f3 = (c + m) * 255, (z + m) * 255, m * 255

                    if 0 <= h < 60:
                        r, g, b = f1, f2, f3
                    elif 60 <= h < 120:
                        r, g, b = f2, f1, f3
                    elif 120 <= h < 180:
                        r, g, b = f3, f1, f2
                    elif 180 <= h < 240:
                        r, g, b = f3, f2, f1
                    elif 240 <= h < 300:
                        r, g, b = f2, f3, f1
                    else:
                        r, g, b = f1, f3, f2

                    rgb_array[x, y][0] = r
                    rgb_array[x, y][1] = g
                    rgb_array[x, y][2] = b

        rgb_image = img.Image()
        rgb_image.data = rgb_array.astype('uint8')
        rgb_image.color_model = ColorModel.rgb
        return rgb_image
