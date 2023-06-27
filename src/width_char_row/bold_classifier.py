from abc import ABC, abstractmethod
from typing import List
from width_char_row.bbox import BBox
import numpy as np
import cv2
from width_char_row.my_binar import binarize

BINARIZE_N = 5
PERMISSIBLE_H_BBOX = 5

TYPE_LINE = 0
TYPE_WORD = 1
TYPE_LINE_WORD = 2

BOLD = 1.0
REGULAR = 0.0


class BaseBoldClassifier(ABC):
    def classify(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        processed_image = self.preprocessing(image)
        lines_estimates = self.get_evaluation_bboxes(processed_image, bboxes)
        lines_bold_indicators = self.clasterization(lines_estimates)
        return lines_bold_indicators

    @abstractmethod
    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        pass

    def get_evaluation_bboxes(self, image: np.ndarray,
                              bboxes: List[List[BBox]]) -> List[List[float]]:
        evaluation_bboxes = []
        for line in bboxes:
            evaluation_bboxes.append([])
            for bbox in line:
                image_bbox = image[bbox.y_top_left:bbox.y_bottom_right,
                             bbox.x_top_left:bbox.x_bottom_right]
                evaluation_bbox = self.evaluation_method(image_bbox)
                evaluation_bboxes[-1].append(evaluation_bbox)
        return evaluation_bboxes

    @abstractmethod
    def evaluation_method(self, image: np.ndarray) -> float:
        pass

    @abstractmethod
    def clasterization(self, lines_estimates: List[List[float]]) -> List[List[float]]:
        pass

    # Функции которые могут понадобиться для реализации других классификаторов ==========
    @staticmethod
    def base_line_image(image: np.ndarray) -> np.ndarray:
        h = image.shape[0]
        if h < PERMISSIBLE_H_BBOX:
            return image
        mean_ = image.mean(1)
        dmean = abs(mean_[:-1] - mean_[1:])

        max1 = 0
        max2 = 0
        argmax1 = 0
        argmax2 = 0
        for i in range(len(dmean)):
            if dmean[i] > max2:
                if dmean[i] > max1:
                    max2 = max1
                    argmax2 = argmax1
                    max1 = dmean[i]
                    argmax1 = i
                else:
                    max2 = dmean[i]
                    argmax2 = i
        h_min = min(argmax1, argmax2)
        h_max = max(argmax1, argmax2)

        return image[h_min:h_max + 1, :]

    @staticmethod
    def get_rid_spaces(image: np.ndarray) -> np.ndarray:
        x = image.mean(0)
        return image[:, x < 0.95]

    # Возращение результатов без кластеризации (полезна при отладке)
    def get_lines_estimates(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        processed_image = self.preprocessing(image)
        lines_estimates = self.get_evaluation_bboxes(processed_image, bboxes)
        return lines_estimates


class PsBoldClassifier(BaseBoldClassifier):
    def __init__(self, k0, type_stat):
        self.k0 = k0
        self.type_stat = type_stat

    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        return binarize(image, BINARIZE_N)

    def clasterization(self, lines_estimates: List[List[float]]) -> List[List[float]]:
        lines_bold_indicators = []

        for i in range(len(lines_estimates)):
            mu = np.mean(lines_estimates[i])
            sigma = np.std(lines_estimates[i])
            lines_bold_indicators.append([])
            if self.type_stat == TYPE_LINE_WORD:
                for j in range(len(lines_estimates[i])):
                    if self.k0 > mu + sigma:
                        lines_bold_indicators[-1].append(BOLD)
                    elif self.k0 < mu - sigma:
                        lines_bold_indicators[-1].append(REGULAR)
                    elif lines_estimates[i][j] > self.k0:
                        lines_bold_indicators[-1].append(REGULAR)
                    else:
                        lines_bold_indicators[-1].append(BOLD)
            elif self.type_stat == TYPE_LINE:
                for j in range(len(lines_estimates[i])):
                    if self.k0 > mu:
                        lines_bold_indicators[-1].append(BOLD)
                    else:
                        lines_bold_indicators[-1].append(REGULAR)
            elif self.type_stat == TYPE_WORD:
                for j in range(len(lines_estimates[i])):
                    if lines_estimates[i][j] > self.k0:
                        lines_bold_indicators[-1].append(REGULAR)
                    else:
                        lines_bold_indicators[-1].append(BOLD)
        return lines_bold_indicators

    def evaluation_method(self, image: np.ndarray) -> float:
        image_p = self.base_line_image(image)
        image_s = self.get_rid_spaces(image_p)
        hw = image_s.shape[0] * image_s.shape[1]
        p_img = image_p[:, :-1] - image_p[:, 1:]
        p_img[p_img > 0] = 1
        p = p_img.sum()
        s = hw - image_s.sum()
        return p / s


class MeanBoldClassifier(BaseBoldClassifier):
    def __init__(self, k0, type_stat):
        self.k0 = k0
        self.type_stat = type_stat

    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        return binarize(image, BINARIZE_N)

    def clasterization(self, lines_estimates: List[List[float]]) -> List[List[float]]:
        lines_bold_indicators = []

        for i in range(len(lines_estimates)):
            mu = np.mean(lines_estimates[i])
            sigma = np.std(lines_estimates[i])
            lines_bold_indicators.append([])
            if self.type_stat == TYPE_LINE_WORD:
                for j in range(len(lines_estimates[i])):
                    if self.k0 > mu + sigma:
                        lines_bold_indicators[-1].append(BOLD)
                    elif self.k0 < mu - sigma:
                        lines_bold_indicators[-1].append(REGULAR)
                    elif lines_estimates[i][j] > self.k0:
                        lines_bold_indicators[-1].append(REGULAR)
                    else:
                        lines_bold_indicators[-1].append(BOLD)
            elif self.type_stat == TYPE_LINE:
                for j in range(len(lines_estimates[i])):
                    if self.k0 > mu:
                        lines_bold_indicators[-1].append(BOLD)
                    else:
                        lines_bold_indicators[-1].append(REGULAR)
            elif self.type_stat == TYPE_WORD:
                for j in range(len(lines_estimates[i])):
                    if lines_estimates[i][j] > self.k0:
                        lines_bold_indicators[-1].append(REGULAR)
                    else:
                        lines_bold_indicators[-1].append(BOLD)
        return lines_bold_indicators

    def evaluation_method(self, image: np.ndarray) -> float:
        bl_image = self.base_line_image(image)
        image_s = self.get_rid_spaces(bl_image)
        return image_s.mean()


