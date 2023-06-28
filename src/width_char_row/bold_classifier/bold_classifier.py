from abc import ABC, abstractmethod
from typing import List
from width_char_row.bbox import BBox
import numpy as np

BOLD = 1.0
REGULAR = 0.0


class BaseBoldClassifier(ABC):
    @abstractmethod
    def classify(self, image: np.ndarray,  bboxes: List[List[BBox]]) -> List[List[float]]:
        pass


