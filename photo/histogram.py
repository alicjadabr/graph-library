import numpy as np
import matplotlib.pyplot as plt


class Histogram:
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, values: np.ndarray = None) -> None:
        if values is None:
            return
        elif values.ndim == 2:
            self.values = np.histogram(values, bins=256, range=(0, 256))[0]
        elif values.ndim == 3:
            red = np.histogram(values[:, :, 0], bins=256, range=(0, 256))[0]
            green = np.histogram(values[:, :, 1], bins=256, range=(0, 256))[0]
            blue = np.histogram(values[:, :, 2], bins=256, range=(0, 256))[0]
            self.values = np.vstack([red, green, blue])

    def plot(self) -> None:
        if self.values.ndim == 1:
            plt.title("Grayscale Histogram")
            plt.xlabel("Grayscale value")
            plt.ylabel("Pixel count")
            plt.plot(self.values)

        elif self.values.ndim == 2:
            plt.figure(figsize=(14, 4))
            plt.subplot(1, 3, 1)
            plt.title("Red Color")
            plt.xlabel("Color value")
            plt.ylabel("Pixel count")
            plt.plot(self.values[:][0], color='r')

            plt.subplot(1, 3, 2)
            plt.title("Green Color")
            plt.plot(self.values[:][1], color='g')

            plt.subplot(1, 3, 3)
            plt.title("Blue Color")
            plt.plot(self.values[:][2], color='b')

    def to_cumulated(self) -> 'Histogram':
        # metoda zwracajaca histogram skumulowany na podstawie stanu wewnetrznego obiektu
        if self.values.ndim == 1:  # dla szarości
            temp_list = []
            sum_h: float = 0
            for i in range(np.size(self.values)):
                sum_h += self.values[i]
                temp_list.append(sum_h)
        else:  # dla kolorowych
            temp_list = [[], [], []]
            for i in range(3):
                sum_h: float = 0
                for j in range(np.size(self.values[:][i])):
                    sum_h += self.values[i][j]
                    temp_list[i].append(sum_h)

        cumulated_array = np.asarray(temp_list)
        cumulated_histogram = Histogram()
        cumulated_histogram.values = cumulated_array
        return cumulated_histogram
