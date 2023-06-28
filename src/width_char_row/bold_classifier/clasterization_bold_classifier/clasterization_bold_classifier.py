from abc import abstractmethod
from ..bold_classifier import BaseBoldClassifier, BOLD, REGULAR
from typing import List
import numpy as np
from width_char_row.bbox import BBox


TYPE_LINE = 0
TYPE_WORD = 1
TYPE_LINE_WORD = 2


class ClasterizationBoldClassifier(BaseBoldClassifier):
    def __init__(self, k0, type_stat):
        self.k0 = k0
        self.type_stat = type_stat

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

    # Возращение результатов без кластеризации (полезна при отладке)
    def get_lines_estimates(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        processed_image = self.preprocessing(image)
        lines_estimates = self.get_evaluation_bboxes(processed_image, bboxes)
        return lines_estimates






