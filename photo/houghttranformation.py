from thresholding import Thresholding
import cv2
import image as img
import numpy as np


class HoughtTransformation(Thresholding):
    def hough_lines(self, thresh:int = 50) -> 'Image':
        lines_img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        _, lines_thresh = cv2.threshold(lines_img,thresh=0,maxval=255,
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lines_edges = cv2.Canny(lines_thresh, 20, 50, 3)
        lines = cv2.HoughLinesP(lines_edges,2,np.pi / 180, threshold=thresh)
        result_lines_img = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2RGB)
        for line in lines:
          x0, y0, x1, y1 = line[0]
          cv2.line(result_lines_img, (x0, y0), (x1, y1), (0, 255, 0), 5)

        image = img.Image()
        image.data = result_lines_img
        return image

    def hough_circles(self, min_dist: int = 10, min_radius: int = 15, max_radius: int = 150) -> 'Image':
        img_bgr = cv2.imread(self.path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_color = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        circles = cv2.HoughCircles(
            img_gray,
            method=cv2.HOUGH_GRADIENT,
            dp=2,
            minDist=min_dist,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        for (x, y, r) in circles.astype(int)[0]:
            cv2.circle(img_color, (x, y), r, (0, 255, 0), 4)

        image = img.Image()
        image.data = img_color
        return image